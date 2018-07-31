import sys
import Configurations
import LinearBlkCodes as lbc
import DataIO
import Iterative_BP_CNN as ibc

# Get configuration
top_cnofig = Configurations.TopConfig()
top_cnofig.parse_cmd_line(sys.argv)

train_config = Configurations.TrainConfig(top_cnofig)

code = lbc.LDPC(top_cnofig.N, top_cnofig.K, top_cnofig.G_file, top_cnofig.H_file)

if top_cnofig.function == 'GenData':
    noise_io = DataIO.NoiseIO(top_cnofig, read_from_file=False)
    intf_io = DataIO.IntfIO(top_cnofig, read_from_file=False)

    # Generate training data
    ibc.generate_noise_samples(code, top_cnofig, train_config, 'Training')


