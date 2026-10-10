"""Bangladesh registration-plate rules: vehicle class <-> plate class letter.

Format (two lines):   <area>[ মেট্রো]-<class letter>      e.g. ঢাকা মেট্রো-গ
                      <series 2 digits>-<number 4 digits>  e.g. ১২-৩৪৫৬
The area (ঢাকা, চট্ট, খুলনা, ...) varies; the class letter is fixed by vehicle category.

SOURCE STATUS (checked 2026-10-10): BRTA does not publish the letter table on brta.gov.bd.
The table below follows the Wikipedia "Vehicle registration plates of Bangladesh" table
(uncited there) cross-checked against Bangla explainer articles (prohori.com 2019, pilabsbd.com).
Where they disagree the entry carries a `conflict` note. Replace with the BRTA gazette when obtained.

All sources agree: ঘ (Gha) = jeep/SUV/crossover. Private sedans/hatchbacks use ক/খ/গ/ভ by engine cc,
so a camera that cannot see engine size gets a SET of letters for "sedan", not a single letter.

These rules CONSTRAIN and VALIDATE an OCR reading; they never write a letter that was not read.
A vehicle-type guess must not be turned into plate evidence (see AGENTS.md / ARCHITECTURE.md).
"""
import re

# letter -> metadata. colour: plate background (white = private, green = commercial/hire).
SERIES_LETTERS = {
    'ক': dict(code='KA',  category='private_car',  detail='car up to ~1000 cc', colour='white',
              conflict='blog: 800 cc'),
    'খ': dict(code='KHA', category='private_car',  detail='car 1001-1300 cc', colour='white'),
    'গ': dict(code='GA',  category='private_car',  detail='car 1301-2000 cc', colour='white',
              conflict='blog: 1500-1800 cc'),
    'ভ': dict(code='BHA', category='private_car',  detail='car above 2000 cc', colour='white'),
    'ঘ': dict(code='GHA', category='suv_jeep',     detail='jeep / SUV / crossover', colour='white'),
    'চ': dict(code='CHA', category='microbus',     detail='microbus / MPV', colour='white'),
    'ছ': dict(code='CHHA', category='human_hauler', detail='human hauler (leguna), hired microbus, ambulance',
              colour='green_or_white'),
    'প': dict(code='PA',  category='taxi',         detail='taxicab', colour='green'),
    'থ': dict(code='THA', category='cng',          detail='CNG auto-rickshaw / baby taxi (hire)', colour='green'),
    'দ': dict(code='DA',  category='cng',          detail='private CNG auto-rickshaw', colour='white',
              conflict='Wikipedia: not described; blog: private CNG'),
    'ত': dict(code='TWA', category='mishuk',       detail='Mishuk auto-rickshaw', colour='green'),
    'ফ': dict(code='FA',  category='tempo',        detail='auto tempo', colour='green'),
    'জ': dict(code='JA',  category='bus',          detail='minibus', colour='green'),
    'ঝ': dict(code='JHA', category='bus',          detail='coach / large bus', colour='white'),
    'ব': dict(code='BA',  category='bus',          detail='inter-city / inter-district bus', colour='green'),
    'স': dict(code='SA',  category='bus',          detail='school / college bus', colour='white'),
    'ঠ': dict(code='THHA', category='pickup',      detail='dual purpose: double-cabin pickup, van', colour='white'),
    'ন': dict(code='NA',  category='pickup',       detail='small pickup', colour='green',
              conflict='Wikipedia: not described'),
    'ম': dict(code='MA',  category='truck',        detail='delivery van / mini truck up to 3.5 t', colour='white'),
    'ট': dict(code='TA',  category='truck',        detail='large truck', colour='green',
              conflict='Wikipedia: not described'),
    'ড': dict(code='DDA', category='truck',        detail='medium truck', colour='green',
              conflict='Wikipedia: not described'),
    'অ': dict(code='AU',  category='truck',        detail='goods 3.5-7.5 t', colour='white'),
    'উ': dict(code='U',   category='truck',        detail='goods 7.5-22 t', colour='white'),
    'ঢ': dict(code='DHA', category='truck',        detail='articulated / tanker / semi-trailer', colour='white_or_green'),
    'শ': dict(code='SHA', category='special',      detail='special purpose: mixer, crane, tow, garbage, hearse', colour='white'),
    'ই': dict(code='E',   category='agricultural', detail='tractor / power tiller', colour='white'),
    'এ': dict(code='A',   category='motorcycle',   detail='motorcycle up to 100 cc', colour='white'),
    'হ': dict(code='HA',  category='motorcycle',   detail='motorcycle 101-125 cc', colour='white',
              conflict='blog: 80-125 cc'),
    'ল': dict(code='LA',  category='motorcycle',   detail='motorcycle above 125 cc', colour='white',
              conflict='Wikipedia 126-165 cc; blog 135-200 cc'),
    'য': dict(code='ZA',  category='any',          detail="Prime Minister's Office, any vehicle", colour='white'),
    'র': dict(code='RA',  category='any',          detail="President's Office, any vehicle", colour='white'),
}

# Visual vehicle types we can (eventually) detect -> categories whose letters are allowed.
VEHICLE_CATEGORIES = {
    'sedan':        ['private_car'],          # also hatchback
    'suv':          ['suv_jeep'],
    'microbus':     ['microbus', 'human_hauler'],
    'leguna':       ['human_hauler'],
    'cng':          ['cng'],
    'mishuk':       ['mishuk'],
    'tempo':        ['tempo'],
    'taxi':         ['taxi'],
    'motorcycle':   ['motorcycle'],
    'bus':          ['bus'],
    'pickup':       ['pickup', 'truck'],      # small pickups registered as ন/ঠ/ম
    'truck':        ['truck'],
    'tractor':      ['agricultural'],
    'special':      ['special'],
    'rickshaw':     [],                       # non-motorised: municipal tag only, no BRTA plate
    'bicycle':      [],
}
# Current COCO vehicle detector classes are too coarse; each maps to the union of possible types.
COCO_TO_TYPES = {
    'car':        ['sedan', 'suv', 'microbus', 'taxi', 'cng', 'pickup'],  # COCO 'car' absorbs CNGs in this footage
    'motorcycle': ['motorcycle'],
    'bus':        ['bus', 'microbus'],   # leguna omitted: no legunas on the current route (user, 2026-10-10)
    'truck':      ['truck', 'pickup', 'cng', 'special'],
    'bicycle':    ['rickshaw'],   # in this footage COCO 'bicycle' is almost always a cycle-rickshaw
}
# Crop-classifier outputs (work/claude/cng-classifier) -> types. 'car' here means "car-like, NOT a CNG".
CLASSIFIER_TO_TYPES = {
    'car':        ['sedan', 'suv', 'microbus', 'taxi', 'pickup'],
    'cng':        ['cng'],
    'motorcycle': ['motorcycle'],
    'bus_truck':  ['bus', 'truck', 'pickup'],
    'rickshaw':   ['rickshaw'],
}
ANY_VEHICLE_LETTERS = {l for l, m in SERIES_LETTERS.items() if m['category'] == 'any'}

def registration_expected(vehicle_type):
    """False for rickshaws/bicycles (no BRTA plate; municipal tags are not registrations), True otherwise."""
    return allowed_letters(vehicle_type) != set()

def vehicle_record(coco_class):
    """How a detected vehicle should be reported. Rickshaws are kept and tracked, but carry no registration."""
    if coco_class == 'bicycle':
        return dict(vehicle_type='rickshaw', registration_expected=False,
                    registration=None, note='rickshaw: no BRTA registration plate')
    return dict(vehicle_type=coco_class, registration_expected=True, registration=None)

def allowed_letters(vehicle_type, include_any=True, source='coco'):
    """Letters a plate on this vehicle type may carry. vehicle_type is one of our types, or a class name
    from the COCO detector (source='coco') or the vehicle crop classifier (source='classifier').
    Returns None for an unknown type (no constraint), set() for vehicles without BRTA plates."""
    table = CLASSIFIER_TO_TYPES if source == 'classifier' else COCO_TO_TYPES
    types = table.get(vehicle_type, [vehicle_type])
    if not all(t in VEHICLE_CATEGORIES for t in types):
        return None
    cats = {c for t in types for c in VEHICLE_CATEGORIES[t]}
    if not cats:
        return set()
    letters = {l for l, m in SERIES_LETTERS.items() if m['category'] in cats}
    return letters | ANY_VEHICLE_LETTERS if include_any else letters

_BN_DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
_PLATE = re.compile(
    r'^\s*(?P<area>[ঀ-৿]+?)(?:\s*(?P<metro>মেট্রো))?\s*[-–—]?\s*'
    r'(?P<letter>[অ-হ])\s*[-–—]?\s*'
    r'(?P<series>\d{2})\s*[-–—]\s*(?P<number>\d{4})\s*$')

def parse_plate(text):
    """Parse 'ঢাকা মেট্রো-গ ১২-৩৪৫৬' (one or two lines, Bangla or ASCII digits). None if not parseable."""
    t = re.sub(r'\s+', ' ', text.translate(_BN_DIGITS).replace('\n', ' '))
    m = _PLATE.match(t)
    if not m or m['letter'] not in SERIES_LETTERS:
        return None
    if not 1 <= int(m['number']) <= 9999:
        return None
    return dict(area=m['area'], metro=bool(m['metro']), letter=m['letter'],
                series=m['series'], number=m['number'], **{'class': SERIES_LETTERS[m['letter']]})

def check_plate(vehicle_type, text):
    """Validate an OCR reading against the vehicle type. Never rewrites the reading."""
    p = parse_plate(text)
    if p is None:
        return dict(status='unparsed', plate=None)
    allowed = allowed_letters(vehicle_type)
    if allowed is None:
        return dict(status='unknown_vehicle_type', plate=p)
    if not allowed:
        return dict(status='vehicle_has_no_brta_plate', plate=p)
    ok = p['letter'] in allowed
    return dict(status='consistent' if ok else 'inconsistent', plate=p, allowed=sorted(allowed))

def constrain_letter(vehicle_type, letter_scores, min_score=0.2):
    """Hard constraint for an OCR letter decoder: keep only letters allowed for the vehicle type.
    letter_scores: {letter: probability}. Returns (letter or None, status).
    Returns None ('abstain') if no allowed letter was actually read with >= min_score;
    it never substitutes a letter that the OCR did not support."""
    allowed = allowed_letters(vehicle_type)
    if allowed is None:
        best = max(letter_scores, key=letter_scores.get, default=None)
        return best, 'unconstrained'
    candidates = {l: s for l, s in letter_scores.items() if l in allowed}
    if not candidates or max(candidates.values()) < min_score:
        return None, 'abstain'
    best = max(candidates, key=candidates.get)
    top = max(letter_scores, key=letter_scores.get)
    return best, 'agrees' if best == top else 'overrode_top_ocr_letter'
