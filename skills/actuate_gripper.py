from bamboo.client import BambooFrankaClient
import sys

client = BambooFrankaClient(server_ip="128.30.224.88")

if sys.argv[1] == "open":
    client.open_gripper()
elif sys.argv[1] == "close":
    client.close_gripper()
else:
    raise NotImplementedError("I can only open or close a gripper")

client.close()
