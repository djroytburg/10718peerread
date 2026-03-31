"""
clean_featurize.py — Data-cleaned version of featurize.py
==========================================================

Implements the following data quality fixes from DATA_PROBLEMS.md:

  F1 (HIGH): Remove author/email metadata from paper content.
      ScienceParse.get_paper_content() concatenates author names and email
      domains into "paper content" used for vocabulary building and hand
      features.  This lets the classifier learn author identity → acceptance
      (point-biserial r = +0.18 to +0.29).  We patch it to use only
      title + abstract + section text.

  R1 (HIGH): Deduplicate reviews.
      100% of ICLR'17 papers have every review exactly duplicated (50%
      duplication ratio, 3 635 / 7 270 entries are copies).  We hash each
      review and keep only the first occurrence.

  R2 (MEDIUM): Remove empty review comments.
      24.6% of ICLR review entries have empty comments — placeholder entries
      from OpenReview, not real reviews.

  R4 (MEDIUM): Filter non-review entries.
      Only 35.8% of ICLR's reviews[] are actual peer reviews.  The rest are
      meta-reviews, committee decisions, author responses, anonymous
      questions, and empty placeholders.  We keep only entries classified as
      'peer_review' or 'reviewer_no_rec' (actual reviewer content).

  F2 (HIGH): Fix hardcoded submission year.
      featurize.py hardcodes 2017 for get_num_recent_references().  For
      ICLR 2017 this is approximately correct, but we derive the year from
      the venue directory name so it generalises to other datasets.

Usage (drop-in replacement for featurize.py):
    python clean_featurize.py <paper-json-dir> <scienceparse-dir> <out-dir> \
        <feature-output-file> <tfidf-vector-file> <max_vocab_size> \
        <encoder> <hand-feature>
"""

import sys, os, random, json, glob, operator, re, hashlib
import pickle as pkl
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from itertools import dropwhile
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from models.Review import Review
from models.Paper import Paper
from models.ScienceParse import ScienceParse
from models.ScienceParseReader import ScienceParseReader


# ======================================================================
# F1 fix — clean paper content (no author names / email domains)
# ======================================================================

def _get_clean_paper_content(sp):
    """Return paper content WITHOUT author names or email domains.

    Original ScienceParse.get_paper_content() does:
        title + abstract + author_names_string + domains_from_emails + sorted sections
    This version drops the author_names_string and domains_from_emails.
    """
    content = sp.title + " " + sp.abstract
    for sect_id in sorted(sp.sections):
        content = content + " " + sp.sections[sect_id]
    # Same page-number stripping as original (P4 — crude but kept for
    # comparability; fixing P4 is low-priority).
    content = re.sub(r"\n([0-9]*\n)+", "\n", content)
    return content


def _monkey_patch_scienceparse():
    """Replace ScienceParse.get_paper_content with the F1-clean version.

    This ensures that all downstream code that calls get_paper_content()
    (hand features like get_num_ref_to_figures, get_num_uniq_words,
    get_avg_sentence_length, get_frequent_words_proportion, etc.)
    automatically uses clean content.
    """
    ScienceParse.get_paper_content = _get_clean_paper_content


# ======================================================================
# R1 + R2 + R4 fixes — review cleaning
# ======================================================================

def _hash_review(rev_obj):
    """Deterministic hash of a Review object for deduplication (R1)."""
    parts = []
    for attr in ['RECOMMENDATION', 'COMMENTS', 'IS_META_REVIEW',
                 'TITLE', 'DATE', 'OTHER_KEYS', 'REVIEWER_CONFIDENCE']:
        val = getattr(rev_obj, attr, None)
        parts.append(f"{attr}={val}")
    blob = "||".join(parts).encode("utf-8", errors="replace")
    return hashlib.sha256(blob).hexdigest()


def _classify_review_type(rev_obj):
    """Classify a Review object into a semantic type (mirrors diagnostics.py).

    Returns one of:
        'peer_review'         — AnonReviewer with RECOMMENDATION, or
                                structured review (ACL/CoNLL)
        'reviewer_no_rec'     — AnonReviewer without RECOMMENDATION but has
                                substantive comments
        'meta_review'         — IS_META_REVIEW is True
        'committee_decision'  — OTHER_KEYS contains 'pcs'
        'empty_reviewer'      — empty comments + no recommendation
        'anonymous_question'  — OTHER_KEYS is '(anonymous)'
        'author_response'     — named person in OTHER_KEYS
        'other'
    """
    is_meta = getattr(rev_obj, 'IS_META_REVIEW', False) or False
    other_keys = str(getattr(rev_obj, 'OTHER_KEYS', '') or '')
    has_rec = rev_obj.RECOMMENDATION is not None
    comments = (rev_obj.COMMENTS or '').strip()
    ok_lower = other_keys.lower()

    if is_meta is True:
        return 'meta_review'
    if 'pcs' in ok_lower:
        return 'committee_decision'
    if 'anonreviewer' in ok_lower:
        if has_rec:
            return 'peer_review'
        if not comments:
            return 'empty_reviewer'
        return 'reviewer_no_rec'
    if '(anonymous)' in other_keys:
        return 'anonymous_question'
    # ACL / CoNLL structured reviews
    if not other_keys and has_rec and comments:
        return 'peer_review'
    if other_keys and other_keys.strip():
        return 'author_response'
    return 'other'


def clean_reviews(paper):
    """Apply R1 (dedup), R2 (empty), R4 (non-review) cleaning in-place.

    Returns a dict of counts for logging.
    """
    original_count = len(paper.REVIEWS)

    # --- R1: deduplicate ---
    seen_hashes = set()
    deduped = []
    r1_removed = 0
    for rev in paper.REVIEWS:
        h = _hash_review(rev)
        if h in seen_hashes:
            r1_removed += 1
            continue
        seen_hashes.add(h)
        deduped.append(rev)

    # --- R2 + R4: keep only actual peer reviews (with or without rec) ---
    # We keep 'peer_review' and 'reviewer_no_rec' — these are real reviewer
    # entries that contain substantive content.
    kept = []
    r2_removed = 0
    r4_removed = 0
    for rev in deduped:
        rtype = _classify_review_type(rev)
        comments = (rev.COMMENTS or '').strip()
        if not comments:
            r2_removed += 1
            continue
        if rtype in ('peer_review', 'reviewer_no_rec'):
            kept.append(rev)
        else:
            r4_removed += 1

    paper.REVIEWS = kept

    return {
        'original': original_count,
        'r1_duplicates_removed': r1_removed,
        'r2_empty_removed': r2_removed,
        'r4_nonreview_removed': r4_removed,
        'final': len(kept),
    }


# ======================================================================
# F2 fix — derive submission year from venue directory path
# ======================================================================

def infer_submission_year(paper_json_dir):
    """Try to infer the submission year from the data directory path.

    Looks for patterns like 'iclr_2017', 'acl_2017', 'conll_2016',
    'arxiv.cs.cl_2007-2017' in the path components.  For range patterns
    (arxiv), returns the end year.  Falls back to 2017 if nothing matches.
    """
    # Walk up path components looking for a year
    parts = os.path.normpath(paper_json_dir).split(os.sep)
    for part in parts:
        # Range pattern: arxiv.cs.cl_2007-2017 → use end year
        m = re.search(r'(\d{4})-(\d{4})', part)
        if m:
            return int(m.group(2))
        # Single year: iclr_2017
        m = re.search(r'(\d{4})', part)
        if m:
            return int(m.group(1))
    return 2017  # fallback


def get_paper_submission_year(paper, default_year):
    """Try to get the submission year for an individual paper.

    Checks DATE_OF_SUBMISSION and histories first, then falls back to
    the venue-level default.
    """
    # DATE_OF_SUBMISSION is present in arxiv papers (e.g. "2016-01-15")
    dos = paper.DATE_OF_SUBMISSION
    if dos and isinstance(dos, str):
        m = re.search(r'(\d{4})', dos)
        if m:
            return int(m.group(1))

    # histories: list of (version, date_str, link, comments)
    if paper.HISTORIES:
        for entry in paper.HISTORIES:
            if entry and len(entry) >= 2 and entry[1]:
                m = re.search(r'(\d{4})', str(entry[1]))
                if m:
                    return int(m.group(1))

    return default_year


# ======================================================================
# Shared helpers (same as featurize.py)
# ======================================================================

def read_features(ifile):
    idToFeature = dict()
    with open(ifile, "rb") as ifh:
        for l in ifh:
            e = l.rstrip().decode("utf-8").split("\t")
            if len(e) == 2:
                idToFeature[e[1]] = e[0]
    return idToFeature


def save_features_to_file(idToFeature, feature_output_file):
    with open(feature_output_file, 'wb') as ofh:
        sorted_items = sorted(idToFeature.items(), key=operator.itemgetter(1))
        for i in sorted_items:
            s = "{}\t{}\n".format(i[1], i[0]).encode("utf-8")
            ofh.write(s)


def save_vect(vect, ofile):
    pkl.dump(vect, open(ofile, "wb"))


def load_vect(ifile):
    return pkl.load(open(ifile, "rb"))


def count_words(corpus, HFW_proportion, most_frequent_words_proportion,
                ignore_infrequent_words_thr):
    counter = Counter(corpus)
    most_common = [x[0] for x in counter.most_common(
        int(len(counter) * HFW_proportion))]
    most_common2 = [x[0] for x in counter.most_common(
        int(len(counter) * (HFW_proportion + most_frequent_words_proportion)))]

    most_frequent_words = set()
    least_frequent_words = set()
    for w in counter:
        if w in most_common2 and w not in most_common:
            most_frequent_words.add(w)
        elif counter[w] < ignore_infrequent_words_thr:
            least_frequent_words.add(w)
    return most_common, most_frequent_words, least_frequent_words


def preprocess(input, only_char=False, lower=False, stop_remove=False,
               stemming=False):
    input = re.sub(r'[^\x00-\x7F]+', ' ', input)
    if lower:
        input = input.lower()
    if only_char:
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(input)
        input = ' '.join(tokens)
    tokens = word_tokenize(input)
    if stop_remove:
        tokens = [w for w in tokens if w not in stopwords.words('english')]
    tokens = [w for w in tokens if len(w) > 1]
    return " ".join(tokens)


# ======================================================================
# Main pipeline
# ======================================================================

def main(args, lower=True, max_vocab_size=False, encoder='bowtfidf'):
    argc = len(args)

    if argc < 9:
        print("Usage:", args[0],
              "<paper-json-dir> <scienceparse-dir> <out-dir>"
              " <feature-output-file> <tfidf-vector-file>"
              " <max_vocab_size> <encoder> <hand-feature>")
        return -1

    paper_json_dir = args[1]
    scienceparse_dir = args[2]
    out_dir = args[3]
    feature_output_file = args[4]
    vect_file = args[5]
    max_vocab_size = False if args[6] == 'False' else int(args[6])
    encoder = False if args[7] == 'False' else str(args[7])
    hand = False if args[8] == 'False' else str(args[8])

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # ------------------------------------------------------------------
    # Apply F1 monkey-patch BEFORE any data loading
    # ------------------------------------------------------------------
    _monkey_patch_scienceparse()
    print("[CLEAN] F1: Patched ScienceParse.get_paper_content "
          "(author/email metadata removed)")

    # ------------------------------------------------------------------
    # F2: Infer submission year from directory path
    # ------------------------------------------------------------------
    venue_year = infer_submission_year(paper_json_dir)
    print(f"[CLEAN] F2: Inferred venue submission year = {venue_year}")

    # ------------------------------------------------------------------
    # Train / test split detection (same logic as original)
    # ------------------------------------------------------------------
    is_train = True
    vect = None
    idToFeature = None
    if os.path.isfile(feature_output_file):
        is_train = False
        idToFeature = read_features(feature_output_file)
        if encoder:
            print('Loading vector file from...', vect_file)
            vect = load_vect(vect_file)
    else:
        print('Loading vector file from scratch..')
        idToFeature = dict()

    outLabelsFile = open(out_dir + '/labels_%s_%s_%s.tsv' % (
        str(max_vocab_size), str(encoder), str(hand)), 'w')
    outIDFile = open(out_dir + '/ids_%s_%s_%s.tsv' % (
        str(max_vocab_size), str(encoder), str(hand)), 'w')
    outSvmLiteFile = open(out_dir + '/features.svmlite_%s_%s_%s.txt' % (
        str(max_vocab_size), str(encoder), str(hand)), 'w')

    # ------------------------------------------------------------------
    # Read papers + apply review cleaning (R1, R2, R4)
    # ------------------------------------------------------------------
    print('Reading reviews from...', paper_json_dir)
    paper_content_corpus = []
    paper_json_filenames = sorted(
        glob.glob('{}/*.json'.format(paper_json_dir)))
    papers = []

    # Aggregate review-cleaning statistics
    total_clean_stats = {
        'original': 0,
        'r1_duplicates_removed': 0,
        'r2_empty_removed': 0,
        'r4_nonreview_removed': 0,
        'final': 0,
    }

    for paper_json_filename in paper_json_filenames:
        paper = Paper.from_json(paper_json_filename)
        paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(
            paper.ID, paper.TITLE, paper.ABSTRACT, scienceparse_dir)

        # get_paper_content() now returns clean content (F1 patched)
        paper_content_corpus.append(
            paper.SCIENCEPARSE.get_paper_content())

        # R1 + R2 + R4: clean reviews
        stats = clean_reviews(paper)
        for k in total_clean_stats:
            total_clean_stats[k] += stats[k]

        papers.append(paper)

    random.shuffle(papers)
    print('Total number of papers:', len(papers))

    # Log review cleaning summary
    print(f"[CLEAN] R1: Removed {total_clean_stats['r1_duplicates_removed']} "
          f"duplicate reviews")
    print(f"[CLEAN] R2: Removed {total_clean_stats['r2_empty_removed']} "
          f"empty reviews")
    print(f"[CLEAN] R4: Removed {total_clean_stats['r4_nonreview_removed']} "
          f"non-review entries (meta-reviews, committee decisions, etc.)")
    print(f"[CLEAN] Reviews: {total_clean_stats['original']} original "
          f"-> {total_clean_stats['final']} kept")

    def get_feature_id(feature):
        if feature in idToFeature:
            return idToFeature[feature]
        else:
            return None

    def addFeatureToDict(fname):
        id = len(idToFeature)
        idToFeature[fname] = id

    # ------------------------------------------------------------------
    # Initialize vocabulary
    # ------------------------------------------------------------------
    outCorpusFilename = out_dir + '/corpus.pkl'
    if not os.path.isfile(outCorpusFilename):
        paper_content_corpus = [
            preprocess(p, only_char=True, lower=True, stop_remove=True)
            for p in paper_content_corpus]
        paper_content_corpus_words = []
        for p in paper_content_corpus:
            paper_content_corpus_words += p.split(' ')
        pkl.dump(paper_content_corpus_words,
                 open(outCorpusFilename, 'wb'))
    else:
        paper_content_corpus_words = pkl.load(
            open(outCorpusFilename, 'rb'))
    print('Total words in corpus', len(paper_content_corpus_words))

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    print('Encoding..', encoder)

    if not encoder:
        print('No encoder', encoder)

    elif encoder in ['bow', 'bowtfidf']:
        word_counter = Counter(paper_content_corpus_words)
        if max_vocab_size:
            word_counter = dict(
                word_counter.most_common()[:max_vocab_size])
        vocabulary = dict()
        for w in word_counter:
            if len(w) and w not in vocabulary:
                if is_train:
                    vocabulary[w] = len(vocabulary)
                    addFeatureToDict(w)
                else:
                    fid = get_feature_id(w)
                    if fid is not None:
                        vocabulary[w] = fid
        print("Got vocab of size", len(vocabulary))
        if is_train:
            print('Saving vectorized', vect_file)
            if encoder == 'bow':
                vect = CountVectorizer(
                    max_df=0.5, analyzer='word',
                    stop_words='english', vocabulary=vocabulary)
            else:
                vect = TfidfVectorizer(
                    sublinear_tf=True, max_df=0.5, analyzer='word',
                    stop_words='english', vocabulary=vocabulary)
            vect.fit([p for p in paper_content_corpus])
            save_vect(vect, vect_file)

    elif encoder in ['w2v', 'w2vtfidf']:
        from sent2vec import (MeanEmbeddingVectorizer,
                              TFIDFEmbeddingVectorizer,
                              import_embeddings)

        word_counter = False
        if max_vocab_size:
            word_counter = Counter(paper_content_corpus_words)
            word_counter = dict(
                word_counter.most_common()[:max_vocab_size])

        if is_train:
            w2v = import_embeddings()
            if encoder == 'w2v':
                vect = MeanEmbeddingVectorizer(w2v, word_counter)
            else:
                vect = TFIDFEmbeddingVectorizer(w2v, word_counter)
            for f in range(vect.dim):
                addFeatureToDict('%s%d' % (encoder, f))
            print('Saving vectorized', vect_file)

            if encoder == 'w2vtfidf':
                vect.fit([p.split() for p in paper_content_corpus])
            save_vect(vect, vect_file)
    else:
        print('Wrong type of encoder', encoder)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build encoding features for all titles
    # ------------------------------------------------------------------
    if encoder:
        all_titles = []
        for p in papers:
            sp = p.get_scienceparse()
            title = p.get_title()
            all_title = preprocess(title, only_char=True, lower=True,
                                   stop_remove=True)
            all_titles.append(all_title)

        if encoder.startswith('w2v'):
            all_titles_features = vect.transform(
                [t.split() for t in all_titles])
        else:
            all_titles_features = vect.transform(all_titles)

    # ------------------------------------------------------------------
    # Register hand features (train only)
    # ------------------------------------------------------------------
    if is_train:
        print('saving features to file', feature_output_file)
        if hand:
            addFeatureToDict("get_most_recent_reference_year")
            addFeatureToDict("get_num_references")
            addFeatureToDict("get_num_refmentions")
            addFeatureToDict("get_avg_length_reference_mention_contexts")
            addFeatureToDict("abstract_contains_deep")
            addFeatureToDict("abstract_contains_neural")
            addFeatureToDict("abstract_contains_embedding")
            addFeatureToDict("abstract_contains_outperform")
            addFeatureToDict("abstract_contains_novel")
            addFeatureToDict("abstract_contains_state_of_the_art")
            addFeatureToDict("abstract_contains_state-of-the-art")

            addFeatureToDict("get_num_recent_references")
            addFeatureToDict("get_num_ref_to_figures")
            addFeatureToDict("get_num_ref_to_tables")
            addFeatureToDict("get_num_ref_to_sections")
            addFeatureToDict("get_num_uniq_words")
            addFeatureToDict("get_num_sections")
            addFeatureToDict("get_avg_sentence_length")
            addFeatureToDict("get_contains_appendix")
            addFeatureToDict("proportion_of_frequent_words")
            addFeatureToDict("get_title_length")
            addFeatureToDict("get_num_authors")

            addFeatureToDict("get_num_ref_to_equations")
            addFeatureToDict("get_num_ref_to_theorems")

        save_features_to_file(idToFeature, feature_output_file)

    # ------------------------------------------------------------------
    # Write features for each paper
    # ------------------------------------------------------------------
    id = 1
    hfws, most_frequent_words, least_frequent_words = count_words(
        paper_content_corpus_words, 0.01, 0.05, 3)

    for p in papers:
        outIDFile.write(str(id) + "\t" + str(p.get_title()) + "\n")
        rec = int(p.get_accepted() == True)
        outLabelsFile.write(str(rec))
        outSvmLiteFile.write(str(rec) + " ")

        sp = p.get_scienceparse()

        # --- Encoder features ---
        if encoder:
            title_tfidf = all_titles_features[id - 1]
            if encoder.startswith('bow'):
                nz = title_tfidf.nonzero()[1]
                for word_id in sorted(nz):
                    outSvmLiteFile.write(
                        str(word_id) + ":" +
                        str(title_tfidf[0, word_id]) + " ")
            elif encoder.startswith('w2v'):
                for word_id in range(vect.dim):
                    outSvmLiteFile.write(
                        str(word_id) + ":" +
                        str(title_tfidf[word_id]) + " ")
            else:
                print('wrong encoder', encoder)
                sys.exit(1)

        # --- Hand features ---
        if hand:
            # F2: use per-paper submission year for recent-references
            paper_year = get_paper_submission_year(p, venue_year)

            outSvmLiteFile.write(
                str(get_feature_id("get_most_recent_reference_year"))
                + ":" +
                str(sp.get_most_recent_reference_year() - 2000) + " ")
            outSvmLiteFile.write(
                str(get_feature_id("get_num_references")) + ":" +
                str(sp.get_num_references()) + " ")
            outSvmLiteFile.write(
                str(get_feature_id("get_num_refmentions")) + ":" +
                str(sp.get_num_refmentions()) + " ")
            outSvmLiteFile.write(
                str(get_feature_id(
                    "get_avg_length_reference_mention_contexts"))
                + ":" +
                str(sp.get_avg_length_reference_mention_contexts())
                + " ")
            outSvmLiteFile.write(
                str(get_feature_id("abstract_contains_deep")) + ":" +
                str(int(p.abstract_contains_a_term("deep"))) + " ")
            outSvmLiteFile.write(
                str(get_feature_id("abstract_contains_neural")) + ":" +
                str(int(p.abstract_contains_a_term("neural"))) + " ")
            outSvmLiteFile.write(
                str(get_feature_id("abstract_contains_embedding"))
                + ":" +
                str(int(p.abstract_contains_a_term("embedding")))
                + " ")
            outSvmLiteFile.write(
                str(get_feature_id("abstract_contains_outperform"))
                + ":" +
                str(int(p.abstract_contains_a_term("outperform")))
                + " ")
            outSvmLiteFile.write(
                str(get_feature_id("abstract_contains_novel")) + ":" +
                str(int(p.abstract_contains_a_term("novel"))) + " ")
            outSvmLiteFile.write(
                str(get_feature_id(
                    "abstract_contains_state_of_the_art"))
                + ":" +
                str(int(p.abstract_contains_a_term(
                    "state of the art")))
                + " ")
            outSvmLiteFile.write(
                str(get_feature_id(
                    "abstract_contains_state-of-the-art"))
                + ":" +
                str(int(p.abstract_contains_a_term(
                    "state-of-the-art")))
                + " ")

            # F2: use paper_year instead of hardcoded 2017
            outSvmLiteFile.write(
                str(get_feature_id("get_num_recent_references"))
                + ":" +
                str(sp.get_num_recent_references(paper_year)) + " ")

            outSvmLiteFile.write(
                str(get_feature_id("get_num_ref_to_figures")) + ":" +
                str(sp.get_num_ref_to_figures()) + " ")
            outSvmLiteFile.write(
                str(get_feature_id("get_num_ref_to_tables")) + ":" +
                str(sp.get_num_ref_to_tables()) + " ")
            outSvmLiteFile.write(
                str(get_feature_id("get_num_ref_to_sections")) + ":" +
                str(sp.get_num_ref_to_sections()) + " ")
            outSvmLiteFile.write(
                str(get_feature_id("get_num_uniq_words")) + ":" +
                str(sp.get_num_uniq_words()) + " ")
            outSvmLiteFile.write(
                str(get_feature_id("get_num_sections")) + ":" +
                str(sp.get_num_sections()) + " ")
            outSvmLiteFile.write(
                str(get_feature_id("get_avg_sentence_length")) + ":" +
                str(sp.get_avg_sentence_length()) + " ")
            outSvmLiteFile.write(
                str(get_feature_id("get_contains_appendix")) + ":" +
                str(sp.get_contains_appendix()) + " ")
            outSvmLiteFile.write(
                str(get_feature_id("proportion_of_frequent_words"))
                + ":" +
                str(round(sp.get_frequent_words_proportion(
                    hfws, most_frequent_words, least_frequent_words),
                    3))
                + " ")
            outSvmLiteFile.write(
                str(get_feature_id("get_title_length")) + ":" +
                str(p.get_title_len()) + " ")
            outSvmLiteFile.write(
                str(get_feature_id("get_num_authors")) + ":" +
                str(sp.get_num_authors()) + " ")
            outSvmLiteFile.write(
                str(get_feature_id("get_num_ref_to_equations")) + ":" +
                str(sp.get_num_ref_to_equations()) + " ")
            outSvmLiteFile.write(
                str(get_feature_id("get_num_ref_to_theorems")) + ":" +
                str(sp.get_num_ref_to_theorems()) + " ")

        outSvmLiteFile.write("\n")
        id += 1

    outLabelsFile.close()
    outIDFile.close()
    outSvmLiteFile.close()
    print('saved', outLabelsFile.name)
    print('saved', outIDFile.name)
    print('saved', outSvmLiteFile.name)


if __name__ == "__main__":
    main(sys.argv)