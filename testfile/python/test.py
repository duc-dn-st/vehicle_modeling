import pandas as pd
import numpy as np
import sys
sys.path.append('.')
import numpy as np
from references.reference import Reference


reference = Reference()

reference.register_reference('references/ovalpath')

print(reference.x)

a = np.full(reference.x.shape[0], 1)

print(a.shape)
