import argparse
from node_classifier import simpleClassifer, train_model, test_model
import torch

EMBED_DIM = 50
RETRAIN = True

def retest(model, noise=0.01):
    print("Hello")
    global generator
    edge_index = None
    if args.noise_type == "erdos":
        data = erdos_dataset(noise)
        data.edge_index = data.edge_index.to('cuda')
        data.x = data.x.to('cuda')
    elif args.noise_type == "power":
        data = powerlaw_dataset(noise)
        data.edge_index = data.edge_index.to('cuda')
        data.x = data.x.to('cuda')
        
    if args.model == "gae":
        embeddings = generator.encode(data.x, data.edge_index).to('cuda')
        
    elif args.model == "dse":
        generator = SpectralEmbedding(data)
        embeddings = generator.get_embeddings().to('cuda')
        
    else:
        edge_index = data.edge_index
        embeddings = data.x
        
    score2 = test_model(model, embeddings[data.test_mask], data.y[data.test_mask], edge_index=edge_index, x = data.x, mask=data.test_mask)
    print(f'Score on noisy data: {score2}')

def main(node_embeddings, data, model = None):
    edge_index = None
    if model is None:
        out_model = simpleClassifer(EMBED_DIM).to('cuda')
    else:
        out_model = model.to('cuda')
        edge_index = data.edge_index
        node_embeddings = data.x
    train_model(out_model, node_embeddings[data.train_mask], data.y[data.train_mask], edge_index=edge_index, x = data.x, mask=data.train_mask)
    score = test_model(out_model, node_embeddings[data.test_mask], data.y[data.test_mask], edge_index=edge_index, x = data.x, mask=data.test_mask)
    # score = test_model(out_model, node_embeddings[data.test_mask], data.y[data.test_mask])
    if not RETRAIN:
        print(f'{args.model}\t{args.noise_type=}\t{args.noise_amount=}\t{data.edges_added=}\t{EMBED_DIM=}\t{score}')
        with open('logs/master.tsv', '+a') as f:
            f.write(f'{args.model}\t{args.noise_type}\t{args.noise_amount}\t{data.edges_added}\t{EMBED_DIM}\t{score}\n')
    else:
        print(f'Score on original noiseless data: {score}')
        retest(out_model)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pump Up The Bass')
    parser.add_argument('model', type=str, help='the model name', choices=["gae", "dse", "bass_line"])
    parser.add_argument('--noise_type', default="erdos", type=str, help='the type of noise to apply', choices=["erdos", "power"])
    parser.add_argument('--noise_amount',default=1e-4, type=float, help='the amount of noise to apply')
    args = parser.parse_args()
    dataset = None
    if args.noise_type == "erdos":
        from dataset import erdos_dataset
        dataset = erdos_dataset(args.noise_amount)
    elif args.noise_type == "power":
        from dataset import powerlaw_dataset
        dataset = powerlaw_dataset(args.noise_amount)
    data = dataset[0]
    embeddings = None
    model = None
    if args.model == "gae":
        from gae import get_gae
        generator = get_gae(data, EMBED_DIM, epochs=50)
        embeddings = generator.encode(data.x, data.edge_index).detach()
        generator = generator.to('cuda')
        print(embeddings.type())
        # exit()
    elif args.model == "dse":
        EMBED_DIM = 600
        from spectral import SpectralEmbedding
        generator = SpectralEmbedding(data)
        embeddings = generator.get_embeddings().to('cuda')
    elif args.model == "bass_line":
        embeddings = None
        from bass_line import GCN
        model = GCN(data.num_node_features, EMBED_DIM)
    main(embeddings, data, model)
