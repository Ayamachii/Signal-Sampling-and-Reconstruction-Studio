import numpy as np
import utils


class ComposerSinusoid:
    def __init__(self, freq, phase, amplitude):
        """
        Initializes a ComposerSinusoid object with the given frequency, phase, and amplitude.

        Parameters:
        freq (float): The frequency of the sinusoid.
        phase (float): The phase of the sinusoid.
        amplitude (float): The amplitude of the sinusoid.
        """

        self.freq = freq
        self.pen_color = utils.generate_random_color()
        self.phase = phase
        self.amplitude = amplitude
        self.continuous_step = 0.005  # 200 Hz
        self.max_range = 10

        formatted_phase = "{:.2f}".format(phase)
        self.label = f"{amplitude} sin( 2 * pi * {freq} * t + {formatted_phase})"

        self.x_values = np.arange(0, self.max_range, self.continuous_step)  # resolution
        self.y_values = self.create_sinusoidal(self.x_values)

    def create_sinusoidal(self, x_values) -> np.ndarray:
        """
        Creates a sinusoidal waveform based on the given x-values.

        Parameters:
        x_values (numpy.ndarray): Array of x-values.
        """
        # Generate sample data (sine wave)
        omega = 2 * np.pi * self.freq
        y_values = self.amplitude * np.sin((omega * x_values) + np.array(self.phase))
        return y_values

    def get_values(self, received_x_values) -> np.ndarray:
        """
        Computes y-values corresponding to the given x-values.

        If the length of received_x_values matches the length of the internal x_values,
        returns the precomputed y_values. Otherwise, computes y_values for the given x_values.

        Parameters:
        received_x_values (numpy.ndarray): Array of x-values.
        """
        if len(received_x_values) == len(self.x_values):  # cont signal
            return self.y_values

        else:
            return self.create_sinusoidal(received_x_values)
