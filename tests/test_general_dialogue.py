from collections import Counter
import unittest

from general_dialogue_spec import (
    AGE_FAMILIES,
    AGE_OCCUPATIONS,
    enumerate_specs,
    load_persona_options,
    load_topic_domains,
)
from gen_dataset_general import (
    PROMPT_VERSION, build_metadata, build_prompt,
    has_forbidden_latin,
    make_blueprint,
    metadata_path,
    normalize_allowed_latin_text,
    normalize_repeated_commas,
    opener_ok,
)
from persona_leak_audit import build_judge_prompt, parse_judgement


class GeneralDialogueSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.personas = load_persona_options()
        cls.domains = load_topic_domains()
        cls.specs = enumerate_specs(120, 4000, cls.personas, cls.domains)

    def test_domain_and_opener_distribution(self):
        by_topic = Counter(spec["topic_id"] for spec in self.specs)
        self.assertEqual(set(by_topic.values()), {120})
        for domain in self.domains:
            counts = Counter(
                spec["opener_idx"] for spec in self.specs
                if spec["topic_id"] == domain.topic_id
            )
            self.assertEqual(set(counts.values()), {120 // len(domain.openers)})

    def test_two_item_pilot_uses_first_two_openers(self):
        specs = enumerate_specs(2, 4000, self.personas, self.domains)
        for domain in self.domains:
            topic_specs = [s for s in specs if s["topic_id"] == domain.topic_id]
            self.assertEqual([s["opener_idx"] for s in topic_specs], [0, 1])
            self.assertEqual([s["opener"] for s in topic_specs], list(domain.openers[:2]))

    def test_persona_compatibility(self):
        for spec in self.specs:
            self.assertIn(spec["occupation"], AGE_OCCUPATIONS[spec["age"]])
            self.assertIn(spec["family"], AGE_FAMILIES[spec["age"]])

    def test_interest_ratio_is_five_five_one(self):
        for domain in self.domains:
            counts = Counter(
                spec["interest"] for spec in self.specs
                if spec["topic_id"] == domain.topic_id
            )
            self.assertLessEqual(abs(counts["乗り気"] - 5 * counts["あまり興味がない"]), 1)
            self.assertLessEqual(abs(counts["普通"] - 5 * counts["あまり興味がない"]), 1)

    def test_fixed_opener_and_allowed_fillers(self):
        spec = self.specs[0]
        self.assertTrue(opener_ok([["A", spec["opener"]]], spec, False))
        with_filler = spec["opener"].replace("こんにちは、", "こんにちは、えーと、")
        self.assertTrue(opener_ok([["A", with_filler]], spec, False))
        self.assertFalse(opener_ok([["A", "こんにちは、別の質問ですか？"]], spec, False))

    def test_greeting_interruption(self):
        spec = self.specs[0]
        rest = spec["opener"][len("こんにちは、"):]
        turns = [["A", "こんにちは、"], ["B", "あ、こんにちは"], ["A", rest]]
        self.assertTrue(opener_ok(turns, spec, True))

    def test_greeting_interruption_rate(self):
        rate = sum(make_blueprint(spec, 44)[0]["greeting_interrupt"] for spec in self.specs)
        rate /= len(self.specs)
        self.assertGreater(rate, 0.25)
        self.assertLess(rate, 0.35)

    def test_t09_latin_exception(self):
        self.assertFalse(has_forbidden_latin("AIとVRとAR", "T09"))
        self.assertFalse(has_forbidden_latin("AIとSNSとYouTube", "T09"))
        self.assertFalse(has_forbidden_latin("ＳＮＳとＹｏｕＴｕｂｅ", "T08"))
        self.assertFalse(has_forbidden_latin("AI", "T08"))
        self.assertFalse(has_forbidden_latin("URLとICTとDIYとTシャツ", "T08"))
        self.assertFalse(
            has_forbidden_latin(
                "AIとITとBGMとビタミンCとDMとCMとAmazonとWebとOSとGoogleとSwitchとRPG",
                "T08",
            )
        )
        self.assertTrue(has_forbidden_latin("T細胞", "T08"))
        self.assertTrue(has_forbidden_latin("C言語", "T08"))

    def test_allowed_latin_is_canonicalized_to_halfwidth(self):
        self.assertEqual(
            normalize_allowed_latin_text("ＳＮＳとｙｏｕｔｕｂｅ", "T07"),
            "SNSとYouTube",
        )
        self.assertEqual(
            normalize_allowed_latin_text("ＡＩとＶＲとＡＲ", "T09"),
            "AIとVRとAR",
        )
        self.assertEqual(
            normalize_allowed_latin_text("ＵＲＬとｉｃｔとｄｉｙとｔシャツ", "T07"),
            "URLとICTとDIYとTシャツ",
        )
        self.assertEqual(
            normalize_allowed_latin_text("ビタミンＣとａｍａｚｏｎとｗｅｂ", "T03"),
            "ビタミンCとAmazonとWeb",
        )

    def test_repeated_commas_are_normalized(self):
        turns = [
            ["A", "うーん、、、そうですね。"],
            ["B", "それは困りますね、、、。"],
        ]
        self.assertEqual(
            normalize_repeated_commas(turns),
            [["A", "うーん、そうですね。"], ["B", "それは困りますね。"]],
        )

    def test_per_dialogue_metadata(self):
        spec = self.specs[0]
        blueprint, _ = make_blueprint(spec, 44)
        metadata = build_metadata(spec, blueprint, {"clean": True})
        _, path = metadata_path("dataset", spec)
        self.assertTrue(path.endswith("metadata/T01/00000/gd_t01_00000.json"))
        self.assertEqual(metadata["persona"]["age"], spec["age"])
        self.assertEqual(metadata["fixed_opener"], spec["opener"])
        self.assertNotIn("topic", metadata)
        self.assertNotIn("topic_domain", metadata)
        self.assertNotIn("topic_index", metadata["sampling"])
        self.assertEqual(metadata["blueprint"], blueprint)
        self.assertEqual(metadata["output"], {"clean": True})

    def test_prompt_keeps_b_persona_hidden_from_a(self):
        spec = self.specs[0]
        blueprint, event_texts = make_blueprint(spec, 44)
        prompt = build_prompt(spec, blueprint, event_texts)
        self.assertIn("A の共有知識ではない", prompt)
        self.assertIn("たとえ推測が人物設定と偶然一致しても不可", prompt)
        self.assertIn("「地元」と言っただけなら地域名は不明", prompt)
        self.assertIn("「仕事」と言っただけなら職種は不明", prompt)
        self.assertIn("general-dialogue-blueprint-v2-no-persona-leak", PROMPT_VERSION)

    def test_prompt_allows_canonical_sns_and_youtube(self):
        spec = self.specs[0]
        blueprint, event_texts = make_blueprint(spec, 44)
        prompt = build_prompt(spec, blueprint, event_texts)
        self.assertIn("Amazon、Web、OS、Google、Switch、RPGだけは表記してよく", prompt)

        t09_spec = next(spec for spec in self.specs if spec["topic_id"] == "T09")
        blueprint, event_texts = make_blueprint(t09_spec, 44)
        prompt = build_prompt(t09_spec, blueprint, event_texts)
        self.assertIn("AI、VR、AR、SNS、YouTube、URL、ICT、DIY、Tシャツ、IT", prompt)

    def test_leak_judge_uses_prefix_and_structured_verdict(self):
        candidate = {
            "field_label": "居住地", "persona_value": "北海道の都市部",
            "a_utterance": "北海道なんですね。",
            "prefix": [["A", "こんにちは。"], ["B", "旅行が好きです。"],
                       ["A", "北海道なんですね。"]],
        }
        prompt = build_judge_prompt(candidate)
        self.assertIn("未来の発話", prompt)
        self.assertIn("北海道の都市部", prompt)
        parsed = parse_judgement(
            '{"verdict":"leak","evidence_turns":[],"confidence":0.9,"reason":"根拠なし"}'
        )
        self.assertEqual(parsed["verdict"], "leak")
        gemma_style = parse_judgement(
            '{"verdict":"grounded","evidence_turns":[B7],"confidence":1.0,"reason":"明示"}'
        )
        self.assertEqual(gemma_style["evidence_turns"], [7])


if __name__ == "__main__":
    unittest.main()
