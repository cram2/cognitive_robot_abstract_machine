from multiverse_client_py import MultiverseClient, MultiverseMetaData


class MultiverseConnector(MultiverseClient):
    def __init__(self, port: str, multiverse_meta_data: MultiverseMetaData) -> None:
        super().__init__(port, multiverse_meta_data)

    def loginfo(self, message: str) -> None:
        print(f"INFO: {message}")

    def logwarn(self, message: str) -> None:
        print(f"WARN: {message}")

    def _run(self) -> None:
        self.loginfo("Start running the client.")
        self._connect_and_start()

    def send_and_receive_meta_data(self) -> None:
        # self.loginfo("Sending request meta data: " + str(self.request_meta_data))
        self._communicate(True)
        # self.loginfo("Received response meta data: " + str(self.response_meta_data))

    def send_and_receive_data(self) -> None:
        # self.loginfo("Sending data: " + str(self.send_data))
        self._communicate(False)
        # self.loginfo("Received data: " + str(self.receive_data))

import time

if __name__ == "__main__":
    multiverse_meta_data = MultiverseMetaData(
        world_name="world",
        simulation_name="simulation",
        length_unit="m",
        angle_unit="rad",
        mass_unit="kg",
        time_unit="s",
        handedness="rhs",
    )
    my_connector = MultiverseConnector(port="5000",
                                       multiverse_meta_data=multiverse_meta_data)
    my_connector.run()

    my_connector.request_meta_data["send"] = {}
    my_connector.request_meta_data["receive"] = {}
    my_connector.request_meta_data["receive"][""] = [""]

    object_data_dict = {}
    my_connector.send_and_receive_meta_data()
    response_meta_data = my_connector.response_meta_data
    for object_name, object_attributes in response_meta_data["receive"].items():
        object_data_dict[object_name] = {}
        for attribute_name, attribute_values in object_attributes.items():
            object_data_dict[object_name][attribute_name] = attribute_values

    while True:
        current_time = time.time()

        sim_time = my_connector.sim_time  # The current simulation time
        my_connector.send_data = [sim_time]
        my_connector.send_and_receive_data()
        receive_data = my_connector.receive_data[1:]
        idx = 0
        for object_name, object_attributes in response_meta_data["receive"].items():
            object_data_dict[object_name] = {}
            for attribute_name, attribute_value in object_attributes.items():
                object_data_dict[object_name][attribute_name] = receive_data[idx:idx + len(attribute_value)]
        world_time = my_connector.world_time

        print(f"Elapsed time: {time.time() - current_time}, Data: {object_data_dict}")


    my_connector.stop()