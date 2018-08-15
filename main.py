import sys
import Configurations
import LinearBlkCodes as lbc
import DataIO
import Iterative_BP_CNN as ibc

# Get configuration
top_cnofig = Configurations.TopConfig()
top_cnofig.parse_cmd_line(sys.argv)

train_config = Configurations.TrainConfig(top_cnofig)
net_config = Configurations.NetConfig(top_cnofig)

code = lbc.LDPC(top_cnofig.N, top_cnofig.K, top_cnofig.G_file, top_cnofig.H_file)

if top_cnofig.function == 'GenData':
    noise_io = DataIO.NoiseIO(top_cnofig, read_from_file=False, noise_file=None)
    intf_io = DataIO.IntfIO(top_cnofig, read_from_file=False, intf_file=None)

    # Generate training data
    ibc.generate_noise_samples(code, top_cnofig, train_config, net_config, 'Training', top_cnofig.BP_iter_nums_gen_data,
                               top_cnofig.currently_trained_net_id, top_cnofig.model_id, noise_io, intf_io)
    ibc.generate_noise_samples(code, top_cnofig, train_config, net_config, 'Test', top_cnofig.BP_iter_nums_gen_data,
                               top_cnofig.currently_trained_net_id, top_cnofig.model_id, noise_io, intf_io)

    print("Finish GenData!")



