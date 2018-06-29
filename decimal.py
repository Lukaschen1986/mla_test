import decimal
from decimal import Decimal
context=decimal.getcontext()
context.rounding = decimal.ROUND_05UP # ROUND_HALF_EVEN 

a = 2.65
b = round(a, 1)
c = round(Decimal(a), 1)
float(c)
