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
    build_metadata,
    has_forbidden_latin,
    make_blueprint,
    metadata_path,
    normalize_repeated_commas,
    opener_ok,
)


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
        self.assertTrue(has_forbidden_latin("AIとSNS", "T09"))
        self.assertTrue(has_forbidden_latin("AI", "T08"))

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


if __name__ == "__main__":
    unittest.main()
