from UI.new_ui import Ui_MainWindow
import pyqtgraph as pg
from composer_sinusoid import ComposerSinusoid
from composed_signal import ComposedSignal
import utils
from typing import Optional
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QMessageBox


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.connect_signal_slots()
        # default is composer
        self.biomedical_flag = False
        self.continuous_step = 0.005  # 200 Hz
        self.max_range = 10
        self.number_of_samples = 1500

        self.components_plotItem = self.ui.Plot_composer.addPlot(
            name="components_plot", row=0, col=0
        )
        self.resultant_plotItem = self.ui.Plot_composer.addPlot(
            name="resultant_plot", row=1, col=0
        )
        self.continuous_signal_and_samples_plot = self.ui.Plot_sampler.addPlot(
            name="continuous_signal_and_samples_plot", row=0, col=0
        )
        self.reconstructed_signal_plot = self.ui.Plot_sampler.addPlot(
            name="reconstructed_signal_plot", row=1, col=0
        )
        self.error_signal_plot = self.ui.Plot_sampler.addPlot(
            name="error_signal_plot", row=2, col=0
        )
        plot_items = [
            self.components_plotItem,
            self.resultant_plotItem,
            self.continuous_signal_and_samples_plot,
            self.reconstructed_signal_plot,
            self.error_signal_plot,
        ]
        for plot_item in plot_items:
            plot_item.setLimits(xMin=0)
            plot_item.setLimits(xMax=self.max_range, xMin=0)
            plot_item.setXRange(0, 5)

        # Additional settings for specific plot items
        self.resultant_plotItem.setXLink("components_plot")
        self.reconstructed_signal_plot.setXLink("continuous_signal_and_samples_plot")
        self.error_signal_plot.setXLink("continuous_signal_and_samples_plot")
        self.error_signal_plot.setYLink("continuous_signal_and_samples_plot")
        self.reconstructed_signal_plot.setYLink("continuous_signal_and_samples_plot")

        self.components_plotItem.addLegend()

        # for a bug: the current sampler object label gets lost when you switch tabs
        self.prev_sampler_label = ""
        # to disable and enable noise check box
        self.not_noise_flag = True
        self.ui.horizontalSlider_snr.setDisabled(self.not_noise_flag)
        # keep track of the previously plotted plotDataItem to be removed before adding the new plotDataItem
        self.sample_markers = None
        self.noise_signal = None
        self.reconstructed_plotDataItem = None
        self.error_plotDataItem = None
        self.current_composed_signal = None
        self.current_sampler_signal = None
        self.error_y_values = []
        # list of composed signals to be called from the sampler
        self.saved_final_composed_signals = []
        # counter to give labels to the composed signals
        self.composed_signal_index = 1
        # value returned from the fsampling slider
        self.fs_slider_value = None

    def connect_signal_slots(self):

        self.ui.btn_addcomp.clicked.connect(self.add_component)
        self.ui.btn_removecomp.clicked.connect(self.remove_component)
        self.ui.btn_save.clicked.connect(self.save)
        self.ui.horizontalSlider_fmax.valueChanged.connect(
            self.check_noise_and_sampling
        )
        self.ui.checkBox.toggled.connect(self.check_noise_and_sampling)
        self.ui.horizontalSlider_snr.valueChanged.connect(self.check_noise_and_sampling)
        self.ui.comboBox_ex.currentTextChanged.connect(self.display_composed_signal)
        self.ui.pushButton.clicked.connect(self.remove_all)
        self.ui.checkBox_2.toggled.connect(self.disable_noise)
        self.ui.actionLoad_Sample.triggered.connect(self.load_sample_function)
        self.ui.actionGet_Example.triggered.connect(self.activate_composer_function)

    def clear_plots_and_reset_slider(self):
        """
        Clears the plots and resets the slider position.
        """
        self.error_signal_plot.clear()
        self.reconstructed_signal_plot.clear()
        self.continuous_signal_and_samples_plot.clear()
        self.reset_slider_position()

    def load_sample_function(self):
        """
        Set up the biomedical data sampling environment.

        Disables certain UI elements, sets up event listeners, retrieves biomedical data,
        and initializes necessary constants and variables.
        """
        # make sure we're in the first tab
        self.ui.tabWidget.setCurrentIndex(0)

        self.clear_plots_and_reset_slider()
        # disable choosing a composed signal, adding noise, the composer tab
        self.biomedical_flag = True
        self.ui.comboBox_ex.setEnabled(False)
        self.ui.checkBox_2.setEnabled(False)
        self.ui.horizontalSlider_snr.setEnabled(False)
        self.ui.tab_2.setEnabled(False)

        # change the slot of moving the slider and toggling normalized checkbox
        self.ui.horizontalSlider_fmax.valueChanged.disconnect()
        self.ui.checkBox.toggled.disconnect()
        self.ui.horizontalSlider_fmax.valueChanged.connect(self.sample_biomedical)
        self.ui.checkBox.toggled.connect(self.sample_biomedical)

        # get the file
        self.biomedical_data = self.get_biomedical_data_the_ys_df()
        # important constants
        self.BIOMEDICAL_MAX_RANGE = 10
        self.BIOMEDICAL_STEP = self.get_biomedical_data_step(
            self.biomedical_data, fs=125
        )  # fs for one column only

        # get x_values, y_values and fmax
        self.biomedical_data_y_values = self.get_y_values_biomedical_data(
            self.biomedical_data
        )

        self.biomedical_data_y_values = self.biomedical_data_y_values[
            : int(self.BIOMEDICAL_MAX_RANGE / self.BIOMEDICAL_STEP)
        ]
        self.biomedical_data_nyquist_fs = self.get_biomedical_fs(self.biomedical_data)
        self.biomedical_data_fmax = 0.5 * self.biomedical_data_nyquist_fs

        print(f"fmax equal{self.biomedical_data_fmax}")
        self.biomedical_data_x_values = self.get_biomedical_data_x_Values(
            self.biomedical_data
        )[: int(self.BIOMEDICAL_MAX_RANGE / self.BIOMEDICAL_STEP)]

        # its color
        self.biomedical_data_pen_color = utils.generate_random_color()
        self.biomedical_data_error_color = utils.generate_random_color()
        self.biomedical_data_reconstructed_color = utils.generate_random_color()

        # now plot the existing ones
        # set the view box of the widget
        self.continuous_signal_and_samples_plot.setXRange(0, 1)

        self.continuous_signal_and_samples_plot.setLimits(
            xMax=self.BIOMEDICAL_MAX_RANGE, xMin=0
        )
        self.reconstructed_signal_plot.setLimits(xMax=self.BIOMEDICAL_MAX_RANGE, xMin=0)

        self.continuous_signal_and_samples_plot.setYRange(
            min(self.biomedical_data_y_values) - 0.1,
            max(self.biomedical_data_y_values) + 0.1,
        )

        # now plot first
        self.sample_biomedical()

    def get_biomedical_data_the_ys_df(self) -> pd.DataFrame:
        """
        Opens a file dialog to select a biomedical data file and returns its contents as a DataFrame.
        """

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            "All Files ();;Text Files (.txt);;CSV Files (*.csv)",
        )
        ecg = pd.read_csv(file_path)

        return ecg

    def get_y_values_biomedical_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extracts the y values from the biomedical data DataFrame.
        """
        if len(data.columns) == 1:
            data_y_values = data.values.flatten()
        elif len(data.columns) >= 2:
            data_y_values = data.iloc[:, -1]
        return data_y_values

    def get_biomedical_fs(self, data: pd.DataFrame) -> float:
        """
        Calculates the sampling frequency (fs) from the biomedical data DataFrame.
        """
        if len(data.columns == 1):
            fs = 1 / self.BIOMEDICAL_STEP
        if len(data.columns) >= 2:
            data_x_values = data.iloc[:, 0].values.flatten()
            fs = 1.0 / (data_x_values[1] - data_x_values[0])
        return fs

    def get_biomedical_data_x_Values(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extracts the x values from the biomedical data DataFrame.
        """
        if len(data.columns) == 1:
            x_values = np.arange(0, self.BIOMEDICAL_MAX_RANGE, self.BIOMEDICAL_STEP)
        elif len(data.columns) >= 2:
            x_values = data.iloc[:, 0].values.flatten()
        return x_values

    def get_biomedical_data_step(self, data: pd.DataFrame, fs: float) -> float:
        """
        Calculates the step (Ts) from the biomedical data DataFrame and the sampling frequency (fs).
        """
        if len(data.columns == 1):
            Ts = 1 / fs
        if len(data.columns) >= 2:
            data_x_values = data.iloc[:, 0].values.flatten()
            Ts = data_x_values[1] - data_x_values[0]
        return Ts

    def activate_composer_function(self) -> None:
        """
        Activates the composer function, enabling composer-related features and resetting plots.

        """
        self.biomedical_flag = False

        self.clear_plots_and_reset_slider()

        # change view range
        self.continuous_signal_and_samples_plot.setXRange(0, 5)
        if self.current_sampler_signal:
            self.reconstructed_signal_plot.setYRange(
                np.min(self.current_sampler_signal.y_values),
                np.max(self.current_sampler_signal.y_values),
            )

        # enable choosing a composed signal, adding noise, the composer tab
        self.ui.comboBox_ex.setEnabled(True)
        self.ui.checkBox_2.setEnabled(True)
        self.ui.horizontalSlider_snr.setEnabled(True)
        self.ui.tab_2.setEnabled(True)

        # call connect signals and slots to call the slot of enabling examples combobox
        self.ui.horizontalSlider_fmax.valueChanged.disconnect()
        self.ui.horizontalSlider_fmax.valueChanged.connect(
            self.check_noise_and_sampling
        )
        self.ui.checkBox.toggled.disconnect()
        self.ui.checkBox.toggled.connect(self.check_noise_and_sampling)

        # maybe remove or return this
        self.check_noise_and_sampling()

    # TODO : reduce repetition
    def sample_biomedical(self):

        self.continuous_signal_and_samples_plot.clear()
        self.continuous_signal_and_samples_plot.plot(
            x=self.biomedical_data_x_values,
            y=self.biomedical_data_y_values,
            pen_color=self.biomedical_data_pen_color,
            name="Biomedical Signal",
        )

        if self.ui.checkBox.isChecked():  # normalized
            self.fs_slider_value = (
                self.ui.horizontalSlider_fmax.value() / 33
            )  # get from slider
            formatted_fs_slider_value = "{:.2f}".format(self.fs_slider_value)
            self.ui.lbl_fmax.setText(f"{formatted_fs_slider_value} fmax")

            self.biomedical_data_x_sampled = (self.biomedical_data)[
                : int(self.BIOMEDICAL_MAX_RANGE / self.BIOMEDICAL_STEP)
            ].resample()

        else:  # unnormalized
            self.fs_slider_value = self.ui.horizontalSlider_fmax.value() * 4
            formatted_fs_slider_value = "{:.2f}".format(self.fs_slider_value)
            self.ui.lbl_fmax.setText(f"{formatted_fs_slider_value} Hz")
            self.biomedical_data_x_sampled = (self.biomedical_data)[
                : int(self.BIOMEDICAL_MAX_RANGE / self.BIOMEDICAL_STEP)
            ].scipy.signal.resample()

        self.biomedical_data_sampled_y_values = self.get_y_values_from_x_values(
            self.biomedical_data_x_values,
            self.biomedical_data_y_values,
            self.biomedical_data_x_sampled,
        )
        if self.sample_markers:
            self.continuous_signal_and_samples_plot.removeItem(self.sample_markers)

        sample_markers = pg.ScatterPlotItem(
            x=self.biomedical_data_x_sampled,
            y=self.biomedical_data_sampled_y_values,
            pen="r",
            symbol="x",
            symbolPen="r",
            symbolBrush="r",
            name="sample_markers",
        )
        self.sample_markers = sample_markers

        self.continuous_signal_and_samples_plot.addItem(sample_markers)
        # interpolation
        self.biomedical_data_reconstructed_y_values = np.interp(
            self.biomedical_data_x_values,
            self.biomedical_data_x_sampled,
            self.biomedical_data_sampled_y_values,
        )
        if self.reconstructed_plotDataItem:
            self.reconstructed_signal_plot.removeItem(self.reconstructed_plotDataItem)

        self.reconstructed_plotDataItem = self.reconstructed_signal_plot.plot(
            x=self.biomedical_data_x_values,
            y=self.biomedical_data_reconstructed_y_values,
            pen=self.biomedical_data_reconstructed_color,
            name="reconstructed_biomedical_signal",
        )

        # error
        self.error_y_values = (
            self.biomedical_data_y_values - self.biomedical_data_reconstructed_y_values
        )

        if self.error_plotDataItem:
            self.error_signal_plot.removeItem(self.error_plotDataItem)

        self.error_plotDataItem = self.error_signal_plot.plot(
            x=self.biomedical_data_x_values,
            y=self.error_y_values,
            pen=self.biomedical_data_error_color,
            name="error_biomedical_signal",
        )

    def disable_noise(self) -> None:
        """
        Toggle the noise flag and disable the signal-to-noise ratio (SNR) slider accordingly.
        """
        self.not_noise_flag = not self.not_noise_flag
        self.ui.horizontalSlider_snr.setDisabled(self.not_noise_flag)
        self.check_noise_and_sampling()

    def remove_component(self) -> None:
        """
        Removes the selected component from the current composed signal and updates the plots accordingly.
        """
        component_label = self.ui.comboBox_comps.currentText()

        if component_label == "Select Component":
            return

        for component in self.current_composed_signal.components:
            if component.label == component_label:
                self.components_plotItem.removeItem(component.plotDataItem)
                self.current_composed_signal.components.remove(component)
                break

        # refreshes the y_values but does not plot the new one
        self.current_composed_signal.y_values = (
            self.current_composed_signal.get_composed_values(
                self.current_composed_signal.x_values
            )
        )

        # refresh fmax of composedSignal
        self.current_composed_signal.fmax = self.current_composed_signal.get_fmax()

        # this is the last component
        if len(self.current_composed_signal.components) == 0:

            for item in self.resultant_plotItem.listDataItems():
                self.resultant_plotItem.removeItem(item)

        else:  # if not the last component

            self.resultant_plotItem.removeItem(
                self.resultant_plotItem.listDataItems()[0]
            )
            # plot the new y_values
            self.resultant_plotItem.plot(
                name=f"composed_sig_{self.current_composed_signal.label}",
                pen=self.current_composed_signal.pen_color,
                x=self.current_composed_signal.x_values,
                y=self.current_composed_signal.y_values,
            )

        self.populate_components_combobox_composer()

    def remove_all(self) -> None:
        """
        Removes all components from the current composed signal and updates the plots accordingly.
        """

        if not self.current_composed_signal.components:
            return

        # remove from components
        for component in self.current_composed_signal.components.copy():
            self.components_plotItem.removeItem(component.plotDataItem)
            self.current_composed_signal.components.remove(component)

        # remove from resultant
        for item in self.resultant_plotItem.listDataItems():
            self.resultant_plotItem.removeItem(item)

        self.populate_components_combobox_composer()

    def reset_slider_position(self) -> None:
        self.ui.horizontalSlider_fmax.setSliderPosition(1)
        self.ui.horizontalSlider_fmax.setMinimum(1)
        self.ui.horizontalSlider_snr.setSliderPosition(99)

    def sinc_interpolation(
        self, y_samples: NDArray, x_samples: NDArray, complete_x: NDArray
    ) -> NDArray:
        """
        Perform sinc interpolation on the given samples.
        """

        if self.ui.checkBox.isChecked():  # normalized

            if self.biomedical_flag:
                Ts = 1 / (self.fs_slider_value * self.biomedical_data_fmax)

            else:
                Ts = 1 / (self.fs_slider_value * self.current_sampler_signal.fmax)
        else:
            Ts = 1 / self.fs_slider_value

        # shifted sinc functions
        sinc_ = np.sinc((complete_x - x_samples[:, None]) / Ts)

        # multiply every sinc by the amplitude of the corresponding sample
        return np.dot(y_samples, sinc_)

    # TODO : reduce repetition
    def check_noise_and_sampling(self):

        if self.not_noise_flag:
            self.sample_without_noise()
        else:
            self.sample_and_add_noise()

    def return_noised(self, x_values:  np.ndarray) -> Optional[np.ndarray]:
        """
        Adds noise to the signal based on the current signal and signal-to-noise ratio (SNR).

        Returns:numpy.ndarray or None: The noised signal if a current sampler signal exists, otherwise None.
        """
        if self.current_sampler_signal is None:
            return

        power = self.current_sampler_signal.y_values**2
        snr_db = (
            self.ui.horizontalSlider_snr.value() / 2.3
        )  # 20 and above good / 10-20 / less than 10 very noisy
        signal_avg_power = np.mean(power)
        signal_avg_power_db = np.log10(signal_avg_power)
        noise_db = signal_avg_power_db - snr_db
        noise_watts = 10 ** (noise_db / 10)
        noise = np.random.normal(0, np.sqrt(noise_watts), len(x_values))

        return self.current_sampler_signal.get_composed_values(x_values) + noise

    # TODO : reduce repetition
    def sample_and_add_noise(self):

        signal_to_be_sampled = self.return_chosen_composed_signal()

        if self.ui.checkBox.isChecked():  # normalized
            self.fs_slider_value = (
                self.ui.horizontalSlider_fmax.value() / 25
            )  # get from slider
            formatted_fs_slider_value = "{:.2f}".format(self.fs_slider_value)
            self.ui.lbl_fmax.setText(f"{formatted_fs_slider_value} fmax")
            x_sampled = np.arange(
                0,
                self.number_of_samples
                / (
                    self.fs_slider_value * signal_to_be_sampled.fmax
                ),  # to get 1000 points
                1 / (self.fs_slider_value * signal_to_be_sampled.fmax),
            )  # 0, 1000/f sampling, 1/f sampling

        else:  # un normalized
            self.fs_slider_value = self.ui.horizontalSlider_fmax.value() / 2
            formatted_fs_slider_value = "{:.2f}".format(self.fs_slider_value)
            self.ui.lbl_fmax.setText(f"{formatted_fs_slider_value} Hz")
            x_sampled = np.linspace(
                0,
                self.number_of_samples / self.fs_slider_value,
                self.number_of_samples + 1,
            )

        signal_to_be_sampled.y_values = self.return_noised(
            self.current_sampler_signal.x_values
        )
        # the sampled y is set inside the get_signal_to_be_sampled function
        signal_to_be_sampled.sampled_x_values = x_sampled
        signal_to_be_sampled.sampled_y_values = self.get_y_values_from_x_values(
            signal_to_be_sampled.x_values,
            signal_to_be_sampled.y_values,
            signal_to_be_sampled.sampled_x_values,
        )
        # signal_to_be_sampled.sampled_y_values = self.return_noised(signal_to_be_sampled.sampled_x_values)

        if self.noise_signal:
            self.continuous_signal_and_samples_plot.removeItem(self.noise_signal)
            pass
        if len(self.continuous_signal_and_samples_plot.listDataItems()):
            self.continuous_signal_and_samples_plot.removeItem(
                self.continuous_signal_and_samples_plot.listDataItems()[0]
            )
            pass

        self.noise_signal = self.continuous_signal_and_samples_plot.plot(
            x=signal_to_be_sampled.x_values,
            y=signal_to_be_sampled.y_values,
            name="noise+signal",
        )

        # noise for reconstruction
        # maybe also add the markers to the reconstructed plot, in this case you HAVE to make another identical item NOT give it the same one(sample_markers)

        # interpolation
        # reconstructed with noise
        signal_to_be_sampled.reconstructed_y_values = self.sinc_interpolation(
            signal_to_be_sampled.sampled_y_values,
            signal_to_be_sampled.sampled_x_values,
            signal_to_be_sampled.x_values,
        )

        if self.reconstructed_plotDataItem:
            self.reconstructed_signal_plot.removeItem(self.reconstructed_plotDataItem)

        self.reconstructed_plotDataItem = self.reconstructed_signal_plot.plot(
            x=signal_to_be_sampled.x_values,
            y=signal_to_be_sampled.reconstructed_y_values,
            pen=signal_to_be_sampled.pen_color,
            name="reconstructed_signal",
        )

        self.error_y_values = (
            signal_to_be_sampled.y_values - signal_to_be_sampled.reconstructed_y_values
        )
        if self.error_plotDataItem:
            self.error_signal_plot.removeItem(self.error_plotDataItem)

        self.error_plotDataItem = self.error_signal_plot.plot(
            x=signal_to_be_sampled.x_values,
            y=self.error_y_values,
            pen=signal_to_be_sampled.pen_color,
            name="error_signal",
        )

        # plot on graph
        if self.sample_markers:
            self.continuous_signal_and_samples_plot.removeItem(self.sample_markers)

        samples_end_at = int(
            self.max_range * signal_to_be_sampled.fmax * self.fs_slider_value
        )
        sample_markers = pg.ScatterPlotItem(
            x=signal_to_be_sampled.sampled_x_values[:samples_end_at],
            y=signal_to_be_sampled.sampled_y_values[:samples_end_at],
            pen="r",
            symbol="x",
            symbolPen="r",
            symbolBrush="r",
            name="sample_markers",
        )
        self.sample_markers = sample_markers

        self.continuous_signal_and_samples_plot.addItem(sample_markers)

    def get_y_values_from_x_values(self, x_values: np.ndarray, y_values: np.ndarray, x_values_to_get: np.ndarray) \
            -> np.ndarray:
        """
        Interpolates y values corresponding to given x values.
        """

        y_values_to_get = []
        y_values_to_get = np.interp(x_values_to_get, x_values, y_values)
        return y_values_to_get

    # TODO : reduce repetition
    def sample_without_noise(self):

        signal_to_be_sampled = self.return_chosen_composed_signal()
        # refresh
        if not signal_to_be_sampled:
            return
        else:
            signal_to_be_sampled.y_values = signal_to_be_sampled.get_composed_values(
                signal_to_be_sampled.x_values
            )

        if (
            signal_to_be_sampled
        ):  # if noise checkbox is toggled (unchecked), refresh continuous plot to remove the noise
            self.refresh_sampler_plots()

        if self.ui.checkBox.isChecked():  # normalized
            self.fs_slider_value = (
                self.ui.horizontalSlider_fmax.value() / 25
            )  # get from slider
            formatted_fs_slider_value = "{:.2f}".format(self.fs_slider_value)
            self.ui.lbl_fmax.setText(f"{formatted_fs_slider_value} fmax")
            x_sampled = np.arange(
                0,
                self.number_of_samples
                / (
                    self.fs_slider_value * signal_to_be_sampled.fmax
                ),  # to get number_of_samples points (about 1500)
                1 / (self.fs_slider_value * signal_to_be_sampled.fmax),
            )

            samples_end_at = int(
                self.max_range * signal_to_be_sampled.fmax * self.fs_slider_value
            )
        else:  # un normalized
            self.fs_slider_value = self.ui.horizontalSlider_fmax.value() / 2
            formatted_fs_slider_value = "{:.2f}".format(self.fs_slider_value)
            self.ui.lbl_fmax.setText(f"{formatted_fs_slider_value} Hz")
            x_sampled = np.linspace(
                0,
                self.number_of_samples / self.fs_slider_value,
                self.number_of_samples + 1,
            )  # again may need tp add plus one

            samples_end_at = int(self.max_range * self.fs_slider_value)

        if not signal_to_be_sampled:
            return

        signal_to_be_sampled.sampled_x_values = x_sampled
        signal_to_be_sampled.sampled_y_values = (
            signal_to_be_sampled.get_composed_values(x_sampled)
        )

        self.refresh_sampler_plots()
        # plot on graph, remove old marks and put new ones
        if self.sample_markers:
            self.continuous_signal_and_samples_plot.removeItem(self.sample_markers)

        sample_markers = pg.ScatterPlotItem(
            x=signal_to_be_sampled.sampled_x_values[:samples_end_at],
            y=signal_to_be_sampled.sampled_y_values[:samples_end_at],
            pen="r",
            symbol="x",
            symbolPen="r",
            symbolBrush="r",
            name="sample_markers",
        )
        self.sample_markers = sample_markers

        self.continuous_signal_and_samples_plot.addItem(sample_markers)

        # interpolation
        signal_to_be_sampled.reconstructed_y_values = self.sinc_interpolation(
            signal_to_be_sampled.sampled_y_values,
            signal_to_be_sampled.sampled_x_values,
            signal_to_be_sampled.x_values,
        )

        if self.reconstructed_plotDataItem:
            self.reconstructed_signal_plot.removeItem(self.reconstructed_plotDataItem)

        self.reconstructed_plotDataItem = self.reconstructed_signal_plot.plot(
            x=signal_to_be_sampled.x_values,
            y=signal_to_be_sampled.reconstructed_y_values,
            pen=signal_to_be_sampled.pen_color,
            name="reconstructed_signal",
        )

        # error
        signal_to_be_sampled.y_values = signal_to_be_sampled.get_composed_values(
            self.current_sampler_signal.x_values
        )
        self.error_y_values = (
            signal_to_be_sampled.y_values - signal_to_be_sampled.reconstructed_y_values
        )

        if self.error_plotDataItem:
            self.error_signal_plot.removeItem(self.error_plotDataItem)

        self.error_plotDataItem = self.error_signal_plot.plot(
            x=signal_to_be_sampled.x_values,
            y=self.error_y_values,
            pen=signal_to_be_sampled.pen_color,
            name="error_signal",
        )

    def display_composed_signal(self):
        """
        Display the selected composed signal in the sampler tab.

        """

        current_sampler_signal_label = self.ui.comboBox_ex.currentText()

        if not current_sampler_signal_label:  # if string was empty
            current_sampler_signal_label = self.prev_sampler_label

        for composed_signal in self.saved_final_composed_signals:
            if composed_signal.label == current_sampler_signal_label:
                self.current_sampler_signal = composed_signal
                break

            elif current_sampler_signal_label == "Examples":
                return

        self.reset_slider_position()

        self.prev_sampler_label = current_sampler_signal_label
        self.refresh_sampler_plots()
        self.check_noise_and_sampling()

    def refresh_sampler_plots(self) -> None:
        """
        Refresh the plots in the sampler tab with the current composed signal.
        """

        if self.current_sampler_signal:

            for item in self.continuous_signal_and_samples_plot.listDataItems():
                self.continuous_signal_and_samples_plot.removeItem(item)

            self.continuous_signal_and_samples_plot.plot(
                x=self.current_sampler_signal.x_values,
                y=self.current_sampler_signal.y_values,
                pen_color=self.current_sampler_signal.pen_color,
                name="composed_signal",
            )

        else:
            self.continuous_signal_and_samples_plot.removeItem(
                self.continuous_signal_and_samples_plot.listDataItems()[0]
            )
            self.reconstructed_signal_plot.removeItem(
                self.reconstructed_signal_plot.listDataItems()[0]
            )
            self.error_signal_plot.removeItem(self.error_signal_plot.listDataItems()[0])

    def return_chosen_composed_signal(self) -> Optional[ComposedSignal]:
        """
        Return the selected composed signal from the examples combobox.

        Returns:
            ComposedSignal or None: The selected composed signal if found, otherwise None.
        """
        current_example = self.ui.comboBox_ex.currentText()
        for composed_signal in self.saved_final_composed_signals:
            if composed_signal.label == current_example:
                return composed_signal
        return None

    def populate_components_combobox_composer(self) -> None:
        """
        Populate the components combo box in the composer interface.
        """
        self.ui.comboBox_comps.clear()
        self.ui.comboBox_comps.addItem("Select Component")
        if len(self.current_composed_signal.components):
            for component in self.current_composed_signal.components:
                self.ui.comboBox_comps.addItem(component.label)

    def populate_components_combobox_Sampler(self) -> None:
        """
        Populate the components combo box in the sampler interface.

        """
        self.ui.comboBox_ex.clear()
        for composed_signal in self.saved_final_composed_signals:
            composed_signal.label = f"ٍSignal_{composed_signal.index}"
            self.ui.comboBox_ex.addItem(composed_signal.label)

    def add_component(self) -> None:
        """
        Add a sinusoidal component to the current composed signal.
        """
        if (
            self.current_composed_signal == None
        ):  # we will need to create a new composed signal object
            new_composed_signal = ComposedSignal()
            self.current_composed_signal = new_composed_signal

        # make Composer Sinusoid object
        amplitude = self.ui.doubleSpinBox_amp.value()
        frequency = self.ui.doubleSpinBox_freq.value()
        phase = self.ui.doubleSpinBox_phase.value()

        # COMPONENT IS NOT ADDED
        new_component = ComposerSinusoid(
            freq=frequency, phase=phase, amplitude=amplitude
        )
        self.current_composed_signal.components.append(new_component)

        plotDataItem = self.components_plotItem.plot(
            name=new_component.label,
            pen=new_component.pen_color,
            x=new_component.x_values,
            y=new_component.y_values,
        )

        new_component.plotDataItem = plotDataItem
        # component done

        # refresh for y_values after adding or removing components
        self.current_composed_signal.y_values = (
            self.current_composed_signal.get_composed_values(
                self.current_composed_signal.x_values
            )
        )

        # refresh f max of composedSignal
        self.current_composed_signal.fmax = self.current_composed_signal.get_fmax()

        # remove old resultant plot if exists
        if len(self.resultant_plotItem.listDataItems()) > 0:  # (1)
            self.resultant_plotItem.removeItem(
                self.resultant_plotItem.listDataItems()[0]
            )

        self.resultant_plotItem.plot(
            name=f"composed_sig_{self.current_composed_signal.label}",
            pen=self.current_composed_signal.pen_color,
            x=self.current_composed_signal.x_values,
            y=self.current_composed_signal.y_values,
        )

        self.populate_components_combobox_composer()

    def save(self) -> None:
        """
        Save the current composed signal.
        """
        if self.current_composed_signal:
            if self.current_composed_signal.components:
                componants_to_be_added = self.current_composed_signal.components.copy()
                new_composed_signal = ComposedSignal(
                    fmax=self.current_composed_signal.fmax,
                    components=componants_to_be_added,
                )
                # we might need to give this new object to the current object
                # check labels of composed signals
                new_composed_signal.index = self.composed_signal_index
                self.composed_signal_index += 1

                self.saved_final_composed_signals.append(new_composed_signal)
                self.populate_components_combobox_Sampler()
                # in the sampler, clear the plots
                self.clear_plots_and_reset_slider()
                QMessageBox.information(
                    self, "Success", "Composed signal has been saved."
                )

            else:
                QMessageBox.warning(self, "Warning", "No composed signal to save.")

        else:
            QMessageBox.warning(self, "Warning", "No composed signal to save.")

