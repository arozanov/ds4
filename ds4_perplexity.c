#include "ds4.h"

/* Perplexity benchmark.
 *
 * Measures cross-entropy of the model on a fixed text, then reports
 * perplexity = exp(NLL/N).  The implementation walks the sequence one
 * token at a time so it can read per-position next-token logits via the
 * public session API.  This is slower than batched prefill but it is the
 * gold-standard quality measurement and matches what llama.cpp's
 * `perplexity` tool reports up to tokenization differences.
 *
 * Intended use: compare vanilla vs TurboQuant KV-cache modes.  Set
 * DS4_TURBO_KV_BITS in the environment to enable TurboQuant; leave it
 * unset for the baseline.
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Hardcoded public-domain text: opening of "Alice's Adventures in
 * Wonderland" by Lewis Carroll (1865).  Long enough to give a stable PPL
 * (~500-700 tokens), short enough to run quickly under token-by-token
 * evaluation.  Tokenization quirks of the DeepSeek tokenizer aside, this
 * text exercises ordinary English prose well. */
static const char *PERPLEXITY_TEXT =
    "Alice was beginning to get very tired of sitting by her sister on the "
    "bank, and of having nothing to do: once or twice she had peeped into "
    "the book her sister was reading, but it had no pictures or "
    "conversations in it, \"and what is the use of a book,\" thought "
    "Alice, \"without pictures or conversations?\" So she was considering "
    "in her own mind (as well as she could, for the hot day made her feel "
    "very sleepy and stupid), whether the pleasure of making a daisy-chain "
    "would be worth the trouble of getting up and picking the daisies, "
    "when suddenly a White Rabbit with pink eyes ran close by her. There "
    "was nothing so very remarkable in that; nor did Alice think it so "
    "very much out of the way to hear the Rabbit say to itself, \"Oh dear! "
    "Oh dear! I shall be too late!\" But when the Rabbit actually took a "
    "watch out of its waistcoat-pocket, and looked at it, and then hurried "
    "on, Alice started to her feet, for it flashed across her mind that "
    "she had never before seen a rabbit with either a waistcoat-pocket, or "
    "a watch to take out of it, and burning with curiosity, she ran across "
    "the field after it, and was just in time to see it pop down a large "
    "rabbit-hole under the hedge. In another moment down went Alice after "
    "it, never once considering how in the world she was to get out again. "
    "The rabbit-hole went straight on like a tunnel for some way, and then "
    "dipped suddenly down, so suddenly that Alice had not a moment to "
    "think about stopping herself before she found herself falling down a "
    "very deep well. Either the well was very deep, or she fell very "
    "slowly, for she had plenty of time as she went down to look about her "
    "and to wonder what was going to happen next. First, she tried to look "
    "down and make out what she was coming to, but it was too dark to see "
    "anything; then she looked at the sides of the well, and noticed that "
    "they were filled with cupboards and book-shelves; here and there she "
    "saw maps and pictures hung upon pegs. She took down a jar from one of "
    "the shelves as she passed; it was labelled \"ORANGE MARMALADE\", but "
    "to her great disappointment it was empty: she did not like to drop "
    "the jar for fear of killing somebody underneath, so managed to put it "
    "into one of the cupboards as she fell past it. \"Well!\" thought "
    "Alice to herself. \"After such a fall as this, I shall think nothing "
    "of tumbling down stairs! How brave they'll all think me at home! "
    "Why, I wouldn't say anything about it, even if I fell off the top of "
    "the house!\" (Which was very likely true.) Down, down, down. Would "
    "the fall never come to an end? \"I wonder how many miles I've fallen "
    "by this time?\" she said aloud. \"I must be getting somewhere near "
    "the centre of the earth. Let me see: that would be four thousand "
    "miles down, I think\" (for, you see, Alice had learnt several things "
    "of this sort in her lessons in the schoolroom, and though this was "
    "not a very good opportunity for showing off her knowledge, as there "
    "was no one to listen to her, still it was good practice to say it "
    "over). \"Yes, that's about the right distance, but then I wonder what "
    "Latitude or Longitude I've got to?\" (Alice had no idea what Latitude "
    "was, or Longitude either, but thought they were nice grand words to "
    "say.) Presently she began again. \"I wonder if I shall fall right "
    "through the earth! How funny it'll seem to come out among the people "
    "that walk with their heads downward!\"";

typedef struct {
    const char *model_path;
    const char *text_path;
    ds4_backend backend;
    int threads;
    int ctx_size;
    bool warm_weights;
    bool quality;
    bool quiet;
} ppl_config;

static double ppl_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4_perplexity [options]\n"
        "\n"
        "Measure model perplexity (PPL) on a fixed text. Set DS4_TURBO_KV_BITS\n"
        "in the environment (3, 4, 6, or 8) to enable the TurboQuant KV cache;\n"
        "leave it unset for the dense baseline.\n"
        "\n"
        "Options:\n"
        "  -m, --model FILE       GGUF model path. Default: ds4flash.gguf\n"
        "  --text FILE            Read text from FILE instead of the built-in snippet.\n"
        "  --ctx N                Allocated context size. Default: 4096\n"
        "  --metal | --cuda | --cpu | --backend NAME\n"
        "                         Select backend. Defaults to Metal on macOS.\n"
        "  -t, --threads N        CPU helper threads.\n"
        "  --quality              Prefer exact kernels where applicable.\n"
        "  --warm-weights         Touch mapped tensor pages before measuring.\n"
        "  --quiet                Suppress per-progress logging.\n"
        "  -h, --help             Show this help.\n");
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "ds4_perplexity: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4_perplexity: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static ds4_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4_perplexity: invalid value for %s: %s\n", opt, s);
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4_perplexity: failed to open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); exit(1); }
    long n = ftell(fp);
    if (n < 0) { fclose(fp); exit(1); }
    if (fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); exit(1); }
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(fp); exit(1); }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) { free(buf); fclose(fp); exit(1); }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static ppl_config parse_options(int argc, char **argv) {
    ppl_config c = {
        .model_path = "ds4flash.gguf",
        .backend = default_backend(),
        .ctx_size = 4096,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--text")) {
            c.text_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--ctx")) {
            c.ctx_size = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = DS4_BACKEND_CUDA;
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else if (!strcmp(arg, "--quiet")) {
            c.quiet = true;
        } else {
            fprintf(stderr, "ds4_perplexity: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }
    return c;
}

int main(int argc, char **argv) {
    ppl_config cfg = parse_options(argc, argv);

    const char *turbo_env = getenv("DS4_TURBO_KV_BITS");
    fprintf(stderr,
            "ds4_perplexity: backend=%s ctx=%d turbo_kv_bits=%s\n",
            ds4_backend_name(cfg.backend),
            cfg.ctx_size,
            turbo_env ? turbo_env : "off");

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) return 1;

    /* Tokenize the input text.  We tokenize as raw text (no chat template)
     * so the resulting PPL is comparable to the standard llama.cpp wikitext
     * methodology: pure next-token cross-entropy. */
    char *owned_text = NULL;
    const char *text = PERPLEXITY_TEXT;
    if (cfg.text_path) {
        owned_text = read_file(cfg.text_path);
        text = owned_text;
    }
    ds4_tokens tokens = {0};
    ds4_tokenize_text(engine, text, &tokens);
    free(owned_text);

    if (tokens.len < 8) {
        fprintf(stderr, "ds4_perplexity: text too short (%d tokens)\n", tokens.len);
        ds4_tokens_free(&tokens);
        ds4_engine_close(engine);
        return 1;
    }
    if (tokens.len + 1 >= cfg.ctx_size) {
        fprintf(stderr,
                "ds4_perplexity: text has %d tokens but ctx is only %d; "
                "increase --ctx or shorten the text\n",
                tokens.len, cfg.ctx_size);
        ds4_tokens_free(&tokens);
        ds4_engine_close(engine);
        return 1;
    }
    fprintf(stderr, "ds4_perplexity: tokenized %d tokens\n", tokens.len);

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg.ctx_size) != 0) {
        fprintf(stderr, "ds4_perplexity: failed to create session\n");
        ds4_tokens_free(&tokens);
        ds4_engine_close(engine);
        return 1;
    }

    /* Sync the session to the first token; this populates s->logits with
     * the next-token distribution P(token_1 | token_0). */
    ds4_tokens prefix = { .v = tokens.v, .len = 1, .cap = 1 };
    char err[256];
    if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4_perplexity: initial sync failed: %s\n", err);
        ds4_session_free(session);
        ds4_tokens_free(&tokens);
        ds4_engine_close(engine);
        return 1;
    }

    /* Walk the sequence one token at a time.  At each step:
     *   1) read logprob of the actual next token from current logits;
     *   2) accumulate negative log likelihood;
     *   3) feed the token to advance the cache and update logits.
     * The first token has no preceding context so it is excluded from the
     * NLL sum, matching the standard PPL convention. */
    const double t0 = ppl_now_sec();
    double nll_sum = 0.0;
    int counted = 0;
    int skipped = 0;
    int last_report = 0;
    for (int i = 1; i < tokens.len; i++) {
        const int target = tokens.v[i];
        ds4_token_score score = {0};
        if (ds4_session_token_logprob(session, target, &score) != 1) {
            fprintf(stderr, "ds4_perplexity: logprob lookup failed at i=%d\n", i);
            ds4_session_free(session);
            ds4_tokens_free(&tokens);
            ds4_engine_close(engine);
            return 1;
        }
        if (isfinite(score.logprob)) {
            nll_sum += -(double)score.logprob;
            counted++;
        } else {
            skipped++;
        }

        if (ds4_session_eval(session, target, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4_perplexity: eval failed at i=%d: %s\n", i, err);
            ds4_session_free(session);
            ds4_tokens_free(&tokens);
            ds4_engine_close(engine);
            return 1;
        }

        if (!cfg.quiet && i - last_report >= 64) {
            const double partial_ppl = counted > 0 ? exp(nll_sum / counted) : 0.0;
            fprintf(stderr,
                    "ds4_perplexity: progress %d/%d  partial_ppl=%.4f\n",
                    i, tokens.len - 1, partial_ppl);
            last_report = i;
        }
    }
    const double t1 = ppl_now_sec();

    if (counted == 0) {
        fprintf(stderr, "ds4_perplexity: no valid tokens scored\n");
        ds4_session_free(session);
        ds4_tokens_free(&tokens);
        ds4_engine_close(engine);
        return 1;
    }

    const double mean_nll = nll_sum / (double)counted;
    const double ppl = exp(mean_nll);

    printf("Tokens: %d\n", counted);
    printf("Negative log likelihood: %.4f\n", nll_sum);
    printf("Mean NLL (cross-entropy): %.4f\n", mean_nll);
    printf("Perplexity: %.4f\n", ppl);
    if (skipped > 0) {
        printf("Skipped non-finite positions: %d\n", skipped);
    }
    fprintf(stderr, "ds4_perplexity: elapsed %.2f s\n", t1 - t0);

    ds4_session_free(session);
    ds4_tokens_free(&tokens);
    ds4_engine_close(engine);
    return 0;
}
