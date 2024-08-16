import torch


def get_vqa_target(example, num_answers, keep_max=False):
    """Convert target classes to tensors with soft scores."""
    target = torch.zeros(num_answers)
    labels = example["target"]["labels"]
    scores = example["target"]["scores"]
    if labels and scores:
        labels = torch.tensor(labels)
        scores = torch.tensor(scores)
        if keep_max:
            labels = labels[scores.argmax()].unsqueeze(0)
            target.scatter_(0, labels, 1.0)
        else:
            target.scatter_(0, labels, scores)
    return target


class VQAMasking:
    """
    Get a mask to separate language and vision tokens.
    """

    def __init__(self, text_fist: bool = True, ignore_cls_tokens: bool = False, ignore_eos_tokens: bool = True) -> None:
        self._text_first = text_fist
        self._ignore_cls_tokens = ignore_cls_tokens
        self._ignore_eos_tokens = ignore_eos_tokens

    def get_lang_mask(self, num_lang_tokens: int, num_vision_tokens: int) -> torch.Tensor:
        lang_mask = torch.zeros(num_lang_tokens + num_vision_tokens, dtype=torch.long)
        if self._text_first:
            start_idx = 0
            end_idx = num_lang_tokens
        else:
            start_idx = num_vision_tokens
            end_idx = num_vision_tokens + num_lang_tokens

        if self._ignore_cls_tokens:
            start_idx += 1
        if self._ignore_eos_tokens:
            end_idx -= 1

        lang_mask[start_idx:end_idx] = 1

        return lang_mask

    def get_image_mask(self, num_lang_tokens: int, num_vision_tokens: int) -> torch.Tensor:
        image_mask = torch.zeros(num_lang_tokens + num_vision_tokens, dtype=torch.long)
        if self._text_first:
            start_idx = num_lang_tokens
            end_idx = num_lang_tokens + num_vision_tokens
        else:
            start_idx = 0
            end_idx = num_vision_tokens

        image_mask[start_idx:end_idx] = 1

        return image_mask

    def get_language_and_image_masks(self, num_lang_tokens: int, num_vision_tokens: int):
        lang_mask = self.get_lang_mask(num_lang_tokens, num_vision_tokens)
        image_mask = self.get_image_mask(num_lang_tokens, num_vision_tokens)

        return lang_mask, image_mask


"""Normalize VQA-v2 answers.

From the official evaluation code at https://github.com/GT-Vision-
Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py.
"""
import re

contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
digit_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]


period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile(r"(\d)(\,)(\d)")
punctuations = [
    ";",
    "/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def normalize_answer(answer: str) -> str:
    """Normalize a VQA answer."""
    answer = answer.replace("\n", " ")
    answer = answer.replace("\t", " ")
    answer = answer.strip()
    answer = process_digit_article(process_punctuation(answer))
    return answer.lower()


def process_punctuation(in_text: str) -> str:
    """Process the answer punctuation."""
    out_text = in_text
    for punct in punctuations:
        punct_cond1 = f"{punct} " in in_text or f" {punct}" in in_text
        punct_cond2 = re.search(comma_strip, in_text) is not None
        if punct_cond1 or punct_cond2:
            out_text = out_text.replace(punct, "")
        else:
            out_text = out_text.replace(punct, " ")
    out_text = period_strip.sub("", out_text, re.UNICODE)
    return out_text


def process_digit_article(in_text: str) -> str:
    """Preprocess digits and articles."""
    out_text = []
    for word in in_text.lower().split():
        word = digit_map.setdefault(word, word)
        if word not in articles:
            out_text.append(word)

    for word_id, word in enumerate(out_text):  # noqa: WPS440
        out_text[word_id] = contractions.get(word, word)
    return " ".join(out_text)
