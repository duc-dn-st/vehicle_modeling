import pandas as pd
import numpy as np


class Reference:
    def __init__(self):
        """! Constructor
        """
        pass

    def register_reference(self, reference_name: str):
        """! Register reference
        """
        data = pd.read_csv(f'{reference_name}.csv')

        for column in data.columns:
            setattr(self, column, data[column].values)

        reference = np.array([getattr(self, column) for column in data.columns]).T

        return reference
