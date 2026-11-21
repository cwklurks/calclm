"""Small task catalog and deterministic allocation for private synthesis runs."""
from collections import Counter, defaultdict
import re


# Labels are task suggestions, not formula implementations or correctness judgments.
SCHOOL_TOPICS = {
    'geometry': '''area of a circle
circumference of a circle
area of a triangle from base and height
triangle area from three sides
area of a trapezoid
area of a parallelogram
area of a rhombus from diagonals
area of an ellipse
area of an annulus
sector area from degrees
arc length from degrees
regular polygon area from apothem
interior angle of a regular polygon
volume of a sphere
surface area of a sphere
volume of a cylinder
total surface area of a cylinder
volume of a cone
slant height of a cone
volume of a pyramid
volume of a conical frustum
diagonal of a rectangular box
distance between two points
midpoint between two points''',
    'algebra': '''slope from two points
point slope line evaluation
line intercept from a point and slope
linear equation with one unknown
quadratic discriminant
quadratic vertex coordinates
sum and product of quadratic roots
evaluate a cubic polynomial
remainder theorem for polynomial division
two by two determinant
solve a two by two linear system
dot product of two dimensional vectors
magnitude of a three dimensional vector
scalar projection of a vector
change of logarithm base
exponential growth after time
exponential decay after time
doubling time from growth rate
half life from decay constant
arithmetic sequence nth term
arithmetic series finite sum
geometric sequence nth term
geometric series finite sum
harmonic mean of two values''',
    'physics': '''constant speed travel time
distance with constant acceleration
final speed with constant acceleration
kinetic energy
gravitational potential energy
momentum from mass and velocity
impulse from force and time
work from force and distance
power from work and time
spring force from extension
elastic spring energy
period of a mass spring oscillator
centripetal acceleration
centripetal force
universal gravitational force
escape speed
Ohm law voltage
electrical power from resistance and current
series resistance
parallel resistance
capacitor stored energy
wave speed from frequency and wavelength
photon energy from wavelength
hydrostatic pressure''',
    'statistics': '''mean of three observations
weighted mean of three observations
population variance of three observations
sample standard deviation of three observations
root mean square of three values
standard error of the mean
z score from observation mean and deviation
confidence interval for a mean with known deviation
coefficient of variation
binomial probability of exactly k successes
binomial expected value and variance
Poisson probability of k events
geometric distribution expected trials
uniform distribution mean and variance
normal distribution interval probability
combinations of n objects taken k
permutations of n objects taken k
probability of independent event intersection
Bayes posterior from priors and likelihoods
linear regression prediction from coefficients''',
    'chemistry': '''moles from mass and molar mass
particles from mole amount
molarity from moles and solution volume
solution dilution final volume
ideal gas pressure
combined gas law final volume
gas density from molar mass
mass percent concentration
parts per million concentration
percent yield from actual and theoretical mass
weighted atomic mass from two isotopes
ph from hydrogen concentration
hydrogen concentration from ph
Henderson Hasselbalch buffer ph
heat from mass specific heat and temperature change
reaction enthalpy from bond energies
first order reaction concentration after time
equilibrium constant from concentrations''',
    'finance': '''simple interest earned
compound interest future value
continuous compounding future value
present value of a future payment
future value of an ordinary annuity
present value of an ordinary annuity
amortized loan periodic payment
loan balance after payments
effective annual interest rate
net present value of three cash flows
straight line depreciation
declining balance depreciation
break even unit sales
profit margin from revenue and cost
price after sequential discounts
sales tax and final price''',
    'conversion': '''Celsius to Fahrenheit temperature
Fahrenheit to Celsius temperature
Celsius to Kelvin temperature
degrees to radians angle
radians to degrees angle
miles to kilometers distance
inches to centimeters length
feet to meters length
pounds to kilograms mass
ounces to grams mass
US gallons to liters volume
square feet to square meters area
cubic inches to cubic centimeters volume
miles per hour to meters per second speed
kilowatt hours to joules energy
psi to pascals pressure
joules to calories energy
decimal hours to hours minutes seconds''',
    'number theory': '''greatest common divisor
least common multiple
Euclidean quotient and remainder
sum of first n natural numbers
sum of first n squares
sum of first n cubes
sum of an integer digit sequence
reverse decimal digits of an integer
count divisors of an integer
sum divisors of an integer
Euler totient by counting coprimes
nth Fibonacci number
triangular number from index
pentagonal number from index
modular exponentiation
integer base digit count''',
    'other': '''derivative of a quadratic at a point
definite integral of a quadratic
trapezoidal integral of a linear function
Simpson integral of a quadratic
average rate of change of a cubic
Newton step for a square root
finite difference derivative of an exponential
polar to Cartesian coordinates
Cartesian to polar radius and angle
two dimensional rotation of a point
sine rule side from two angles and a side
cosine rule side from two sides and angle
right triangle hypotenuse from legs
right triangle angle from opposite and adjacent
percent change between values
weighted course grade''',
}


def _words(label):
    return set(re.findall(r'[a-z0-9]+', label.lower())) - {
        'a', 'an', 'and', 'at', 'by', 'for', 'from', 'in', 'of', 'the', 'to',
        'is', 'enter', 'input', 'output', 'result', 'answer', 'number', 'value',
    }


def build_catalog(seeds):
    """School tasks plus up to 24 usable seed titles. Seed titles stay private."""
    from lab.synth import program_title
    from lab.mine_math import TOPIC_WORDS

    catalog = [{'topic': topic, 'label': label}
               for topic, labels in SCHOOL_TOPICS.items() for label in labels.splitlines()]
    seen = {frozenset(_words(row['label'])) for row in catalog}
    extras = {}
    task_word = re.compile(r'\b(?:' + '|'.join(TOPIC_WORDS.values()) + r')\b', re.I)
    for row in seeds:
        title = program_title(row['text'])
        if (not title or not 6 <= len(title) <= 55 or
                not re.fullmatch(r'[A-Za-z0-9 ()/.,%+*^=\-]+', title) or
                len([w for w in _words(title) if len(w) >= 3]) < 2 or
                re.search(r'\b(took|getting|calc|express|ti|prgm|program|ch|about)\b',
                          title, re.I) or not task_word.search(title)):
            continue
        key = frozenset(_words(title))
        if key not in seen:
            extras.setdefault(key, {'topic': row['topic'], 'label': title})
    catalog.extend(sorted(extras.values(), key=lambda row: (row['topic'], row['label']))[:24])
    return catalog


class TopicSteering:
    def __init__(self, catalog, kept=(), attempts=()):
        self.catalog = catalog
        self.words = {row['label']: _words(row['label']) for row in catalog}
        self.coverage, self.assigned = Counter(), Counter()
        self.titles = defaultdict(list)
        for row in kept:
            self.keep(row)
        for row in attempts:
            if row.get('subtopic') in self.words:
                self.assigned[row['subtopic']] += 1

    def _label(self, row):
        if row.get('subtopic') in self.words:
            return row['subtopic']
        words = _words(row.get('title') or '')
        # Legacy labels are rough coverage hints; do not claim semantic classification.
        scores = [(len(words & terms) / len(words | terms), label)
                  for label, terms in self.words.items() if len(words & terms) >= 2]
        score, label = max(scores, default=(0, None))
        return label if score >= 0.5 else None

    def keep(self, row):
        label = self._label(row)
        if label is not None:
            self.coverage[label] += 1
            if row.get('title') and row['title'] not in self.titles[label]:
                self.titles[label].append(row['title'])

    def covered_titles(self, label):
        return self.titles[label][-40:]

    def assign(self):
        # Four assignments weigh as much as one kept program. Count submissions now so
        # a 600-call batch rotates, and a persistently failing topic cannot monopolize it.
        row = min(self.catalog, key=lambda row: (
            4 * self.coverage[row['label']] + self.assigned[row['label']],
            self.assigned[row['label']]))
        self.assigned[row['label']] += 1
        return row
