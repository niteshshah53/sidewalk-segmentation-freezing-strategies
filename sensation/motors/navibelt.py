"""The control for the navibelt to use it in SENSATION syste."""

import time
from pybelt.examples_utility import interactive_belt_connection
from pybelt.belt_controller import (
    BeltController,
    BeltConnectionState,
    BeltControllerDelegate,
    BeltMode,
    BeltOrientationType,
    BeltVibrationTimerOption,
)


class Delegate(BeltControllerDelegate):
    # Belt controller delegate
    pass


class NaviBelt:
    def __init__(self):
        self.direction_channel = 0
        self.sleep_time = 1
        self.drift_channel = 1
        self.connect()

    def connect(self):
        print("Starting interactive connection to navibelt...")
        # Interactive script to connect the belt
        belt_controller_delegate = Delegate()
        belt_controller = BeltController(belt_controller_delegate)
        interactive_belt_connection(belt_controller)
        if belt_controller.get_connection_state() != BeltConnectionState.CONNECTED:
            print("Connection failed.")
            return 0

        # Change belt mode to APP mode
        belt_controller.set_belt_mode(BeltMode.APP_MODE, wait_ack=True)

        # Stop orientation warning signal
        belt_controller.set_inaccurate_orientation_signal_state(
            enable_in_app=False,
            save_on_belt=False,
            enable_in_compass=False,
            wait_ack=True,
        )

        self.belt_controller = belt_controller

    def send_pulse_by_angle(self, channel: int, angle: int):
        """Send pulse by angle"""
        if self.check_connection():
            self.belt_controller.send_pulse_command(
                channel_index=channel,
                orientation_type=BeltOrientationType.ANGLE,
                orientation=angle,
                intensity=None,
                on_duration_ms=150,
                pulse_period=500,
                pulse_iterations=5,
                series_period=1500,
                series_iterations=1,
                timer_option=BeltVibrationTimerOption.RESET_TIMER,
                exclusive_channel=False,
                clear_other_channels=False,
            )

    def send_pulse_by_motor_index(
        self, channel: int, motor_index: int, iterations: int = 3
    ):
        """Send pulse by motor index.

        Motor index needs to transformet to belt index.
        Belt motor index 0 starts front center.
        SENSATION starts left side with 0.
        """
        if self.check_connection():
            self.belt_controller.send_pulse_command(
                channel_index=channel,
                orientation_type=BeltOrientationType.MOTOR_INDEX,
                orientation=motor_index,
                intensity=None,
                on_duration_ms=150,
                pulse_period=500,
                pulse_iterations=1,
                series_period=1000,
                series_iterations=iterations,
                timer_option=BeltVibrationTimerOption.RESET_TIMER,
                exclusive_channel=False,
                clear_other_channels=False,
            )
            time.sleep(self.sleep_time)

    def check_connection(self):
        if self.belt_controller.get_connection_state() == BeltConnectionState.CONNECTED:
            return True
        else:
            return False

    def stay_center(self):
        """Activates center motor to indicate stay center"""
        self.send_pulse_by_motor_index(self.direction_channel, 0, 2)

    def go_right(self):
        """Activate motor to indicate go right"""
        self.send_pulse_by_motor_index(self.direction_channel, 4, 2)

    def go_left(self):
        """Indicates go left with left motor"""
        self.send_pulse_by_motor_index(self.direction_channel, 11, 2)

    def map_motor_index(self, idx):
        """Maps digits from 1 .. 8 to motor index: 12, 13, 14, 15, 0, 1, 2, 3, 4"""
        mapping = {1: 12, 2: 13, 3: 14, 4: 15, 5: 0, 6: 1, 7: 2, 8: 3}

        return mapping.get(idx, None)

    def indicate_drift(self, idx):
        """Use this to indicate drift by using motors in the front"""
        # Get the coresponding motor
        motor_idx = self.map_motor_index(idx)

        self.send_pulse_by_motor_index(self.drift_channel, motor_idx, 2)

    def disconnect(self):
        self.belt_controller.disconnect_belt()
