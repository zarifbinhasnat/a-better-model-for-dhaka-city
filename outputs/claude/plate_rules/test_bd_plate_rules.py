"""Run: python3 outputs/claude/plate_rules/test_bd_plate_rules.py"""
import unittest
from bd_plate_rules import allowed_letters, parse_plate, check_plate, constrain_letter, registration_expected, vehicle_record

class PlateRules(unittest.TestCase):
    def test_parse_two_line_bangla(self):
        p = parse_plate('ঢাকা মেট্রো-গ\n১২-৩৪৫৬')
        self.assertEqual((p['area'], p['metro'], p['letter'], p['series'], p['number']),
                         ('ঢাকা', True, 'গ', '12', '3456'))

    def test_area_varies_letter_fixed(self):
        p = parse_plate('চট্ট মেট্রো-ঘ ১১-০০০১')
        self.assertEqual((p['area'], p['letter'], p['class']['category']), ('চট্ট', 'ঘ', 'suv_jeep'))

    def test_district_without_metro(self):
        self.assertFalse(parse_plate('কুমিল্লা-হ ৫১-২২৩৩')['metro'])

    def test_rejects_garbage_and_zero_number(self):
        self.assertIsNone(parse_plate('UTTARA'))
        self.assertIsNone(parse_plate('ঢাকা মেট্রো-গ ১২-০০০০'))

    def test_sedan_is_not_gha(self):
        self.assertEqual(allowed_letters('sedan', include_any=False), {'ক', 'খ', 'গ', 'ভ'})
        self.assertEqual(check_plate('sedan', 'ঢাকা মেট্রো-ঘ ১২-৩৪৫৬')['status'], 'inconsistent')
        self.assertEqual(check_plate('suv', 'ঢাকা মেট্রো-ঘ ১২-৩৪৫৬')['status'], 'consistent')

    def test_cng_leguna_motorcycle_rickshaw(self):
        self.assertEqual(allowed_letters('cng', include_any=False), {'থ', 'দ'})
        self.assertIn('ছ', allowed_letters('leguna'))
        self.assertEqual(allowed_letters('motorcycle', include_any=False), {'এ', 'হ', 'ল'})
        self.assertEqual(allowed_letters('rickshaw'), set())
        self.assertEqual(check_plate('rickshaw', 'ঢাকা মেট্রো-থ ১১-২২২২')['status'], 'vehicle_has_no_brta_plate')

    def test_coco_car_is_weak_constraint(self):
        a = allowed_letters('car')
        self.assertTrue({'গ', 'ঘ', 'চ', 'থ'} <= a)   # COCO 'car' includes CNGs and SUVs in this footage

    def test_unknown_type_is_unconstrained(self):
        self.assertIsNone(allowed_letters('spaceship'))
        self.assertEqual(constrain_letter('spaceship', {'গ': .9})[1], 'unconstrained')

    def test_constrain_never_invents(self):
        # OCR reads ঘ strongly, গ weakly: on a sedan the constraint picks গ only because OCR supported it
        self.assertEqual(constrain_letter('sedan', {'ঘ': .7, 'গ': .25}), ('গ', 'overrode_top_ocr_letter'))
        # no allowed letter was read -> abstain, never output a made-up letter
        self.assertEqual(constrain_letter('sedan', {'ঘ': .95, 'গ': .02}), (None, 'abstain'))
        self.assertEqual(constrain_letter('cng', {'থ': .8, 'ঘ': .1}), ('থ', 'agrees'))

    def test_rickshaw_tracked_without_registration(self):
        r = vehicle_record('bicycle')
        self.assertEqual((r['vehicle_type'], r['registration_expected'], r['registration']), ('rickshaw', False, None))
        self.assertFalse(registration_expected('bicycle'))
        self.assertFalse(registration_expected('rickshaw'))
        self.assertTrue(registration_expected('motorcycle'))
        self.assertTrue(vehicle_record('car')['registration_expected'])

if __name__ == '__main__':
    unittest.main()
