import numpy as np
import utils


class ComposedSignal:
    def __init__(self, fmax=-1, components=[], label=""):
        """
        Initializes a ComposedSignal object with the given maximum frequency, components, and label.

        Parameters:
        fmax (float, optional): The maximum frequency among the components.
        components (list, optional): List of ComposerSinusoid objects representing individual sinusoidal components.
        label (str, optional): A label describing the composed signal.
        """
        self.components = components
        self.fmax = fmax
        self.pen_color = utils.generate_random_color()
        self.label = label
        self.continuous_step = 0.005  # 200 Hz
        self.max_range = 10
        self.x_values = np.arange(0, self.max_range, self.continuous_step)
        self.y_values = self.get_composed_values(self.x_values)

    def get_fmax(self) -> float:
        """
        Computes the maximum frequency among the components.
        """
        if not self.components:
            return -1
        else:
            self.fmax = -1
            for component in self.components:
                if component.freq > self.fmax:
                    self.fmax = component.freq

            return self.fmax

    def get_composed_values(self, received_x_values) -> np.ndarray:
        """
        Computes the composed signal by summing the individual components.

        Parameters:
        received_x_values (numpy.ndarray): Array of x-values.
        """
        composed_signal = np.zeros(len(received_x_values))
        for signal in self.components:
            composed_signal += signal.get_values(received_x_values)

        return composed_signal
