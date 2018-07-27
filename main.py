import sys
import Configurations
import LinearBlkCodes as lbc
import DataIO

# Get configuration
top_cnofig = Configurations.TopConfig()
top_cnofig.parse_cmd_line(sys.argv)

code = lbc.LDPC(top_cnofig.N, top_cnofig.K, top_cnofig.G_file, top_cnofig.H_file)

if top_cnofig.function == 'GenData':
    noise_io = DataIO.NoiseIO(top_cnofig, read_from_file=False)
    intf_io = DataIO.IntfIO(top_cnofig, read_from_file=False)

