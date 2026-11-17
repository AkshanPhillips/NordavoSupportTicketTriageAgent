import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import redact_payment_details


class TestPaymentRedaction(unittest.TestCase):
    def test_redacts_dutch_iban(self):
        sample = "Mijn rekeningnummer is NL91 ABNA 0417 1643 00 voor de terugstorting."
        redacted = redact_payment_details(sample)
        self.assertNotIn("NL91", redacted)
        self.assertNotIn("0417", redacted)
        self.assertIn("[REDACTED_IBAN]", redacted)

    def test_redacts_compact_iban(self):
        sample = "Betaling via NL12INGB0001234567 ontvangen."
        redacted = redact_payment_details(sample)
        self.assertNotIn("NL12INGB", redacted)
        self.assertIn("[REDACTED_IBAN]", redacted)

    def test_redacts_credit_card(self):
        sample = "Betaald met creditcard 4532-1234-5678-9012."
        redacted = redact_payment_details(sample)
        self.assertNotIn("4532", redacted)
        self.assertIn("[REDACTED_CARD]", redacted)

    def test_clean_text_unchanged(self):
        sample = "Hoe voeg ik een nieuwe medewerker toe aan Team?"
        self.assertEqual(redact_payment_details(sample), sample)


if __name__ == "__main__":
    unittest.main()
