import argparse
from node_classifier import simpleClassifer, train_model, test_model
import torch

EMBED_DIM = 600

def main(node_embeddings, data):
    out_model = simpleClassifer(EMBED_DIM).to('cuda')
    train_model(out_model, node_embeddings[data.train_mask], data.y[data.train_mask])
    score = test_model(out_model, node_embeddings[data.train_mask], data.y[data.train_mask])
    score = test_model(out_model, node_embeddings[data.test_mask], data.y[data.test_mask])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pump Up The Bass')
    parser.add_argument('model', type=str, help='the model name', choices=["gae", "dse"])
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
    if args.model == "gae":
        from gae import get_gae
        model = get_gae(data, EMBED_DIM, epochs=50)
        embeddings = model.encode(data.x, data.edge_index).detach()
        print(embeddings.type())
        # exit()
    elif args.model == "dse":
        from spectral import SpectralEmbedding
        model = SpectralEmbedding(data)
        embeddings = model.get_embeddings().to('cuda')
    main(embeddings, data)

    
