from collections import defaultdict
import re
from .helpers import lemmatization
from .proper_check import check_if_proper
from .typos_final_filter import refine_typos
import string
import logging

logger = logging.getLogger(__name__)

def check_exeptions(matches, blocks, proper_names, main_font):
    '''Filters raw language-tool matches to drop false positives and keep actual errors, passes 
    typing errors for secondary check.'''
    logger.info("Starting exception filtering: %d raw matches to filter", len(matches))
    potential_exeptions = defaultdict(list)
    valid_errors = []
    blocks_to_check = defaultdict(list)
    for match in matches:
        blocks_to_check[match.block_id].append(match)
    for block in blocks:
        if not block.block.words:
            continue
        block_key = block.block.block_id
        if block_key in blocks_to_check:
            for match in blocks_to_check[block_key]:
                text = block.contents
                word = match.content
                word_clean = word.strip(string.punctuation + string.whitespace)
                if word_clean in {p[0] for p in proper_names}:
                    continue
                elif word_clean.lower() in {p[1] for p in proper_names}:
                    continue
                potential_exeption = False
                inside_quotes = check_quotes(match.offset, match.offset + match.error_length, text)
                if not inside_quotes:
                    if match.category in {"TYPOS", "SPELLING" ,"COMPOUNDING", "SYNTAX"}:
                        if remove_errors_different_font(match, block, main_font):
                            continue
                        try:
                            if any(letter == '-' for letter in match.content):
                                continue
                            elif match.offset > 0 and text[match.offset - 1] == '-':
                                continue
                            elif match.offset + len(match.content) < len(text) and text[match.offset + len(match.content)] == '-':
                                continue
                        except Exception:
                            logger.exception("Failed to check adjacent characters for dash")
                        lemma = lemmatization(word, block.language)
                        if check_if_proper(block.block, match, proper_names, lemma):
                            continue
                        potential_exeptions[lemma].append(match)
                        potential_exeption = True
                    if match.category == "CASING" and match.offset == 0:
                        continue
                if not inside_quotes and not potential_exeption:
                    valid_errors.append(match)
    exeptions = []  
    before_typo_refine = len(valid_errors)
    valid_errors = refine_typos(valid_errors, blocks) #fallback od redakcji
    after_typo_refine = len(valid_errors)
    repeated_words_dropped = 0
    for lemma, match_list in potential_exeptions.items():
        # if len(match_list) > 2:
        #     exeptions.extend(match_list)
        #     repeated_words_dropped += len(match_list)
        # else:
        #     valid_errors.extend(match_list)
        valid_errors.extend(match_list)

    logger.info("Exception filtering complete:")
    logger.info("  Input: %d, output: %d", len(matches), len(valid_errors))
    logger.info("  Potential typos checked: %d unique lemmas", len(potential_exeptions))
    logger.info("  Repeated words dropped (>2 occurrences): %d", repeated_words_dropped)
    logger.info("  Typo refinement: %d -> %d", before_typo_refine, after_typo_refine)
    return valid_errors


def remove_errors_different_font(match, block, main_font):
    '''Exclueds matches that use different font than main tex for example code snippet citations.'''
    if not main_font or not match.word_idxs:
        return False
    words_by_idx = {w.word_index: w for w in block.block.words}
    for idx in match.word_idxs:
        word = words_by_idx.get(idx)
        if word is not None and word.font != main_font:
            return True
    return False

def check_quotes(start, end, text, return_spans=False):
    '''Looks for quotes fragments in text and returns their spans, to exclude them from certain error types.'''
    spans = []
    for quote_match in re.finditer(r'[“„″””\'”"](.+?)[“”″\'"”]', text):
        if return_spans:
            spans.append((quote_match.start(), quote_match.end()))
        elif quote_match.start() < start < quote_match.end() or quote_match.start() < end < quote_match.end():
            return True
    if return_spans:
        return spans
    return False



