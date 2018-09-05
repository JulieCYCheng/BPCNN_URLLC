import sys
import Configurations
import LinearBlkCodes as lbc
import DataIO
import Iterative_BP_CNN as ibc
import ConvNet

# Get configuration
top_config = Configurations.TopConfig()
top_config.parse_cmd_line(sys.argv)

train_config = Configurations.TrainConfig(top_config)
net_config = Configurations.NetConfig(top_config)

code = lbc.LDPC(top_config.N, top_config.K, top_config.G_file, top_config.H_file)

if top_config.function == 'GenData':
    noise_io = DataIO.NoiseIO(top_config, read_from_file=False, noise_file=None)
    intf_io = DataIO.IntfIO(top_config, read_from_file=False, intf_file=None)

    # Generate training data
    ibc.generate_noise_samples(code, top_config, train_config, net_config, 'Training', top_config.BP_iter_nums_gen_data,
                               top_config.currently_trained_net_id, top_config.model_id, noise_io, intf_io)
    ibc.generate_noise_samples(code, top_config, train_config, net_config, 'Test', top_config.BP_iter_nums_gen_data,
                               top_config.currently_trained_net_id, top_config.model_id, noise_io, intf_io)

    print("Finish GenData!")

elif top_config.function == 'Train':
    net_id = top_config.currently_trained_net_id
    conv_net = ConvNet.ConvNet(net_config, train_config, net_id)
    conv_net.train_network(top_config.model_id)

    print("Finish Train!")
