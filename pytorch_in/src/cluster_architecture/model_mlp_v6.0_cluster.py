import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

SERIE = sys.argv[1]
EPOCHS = 3000
input_size = 1
output_size = 1

hidden_layers = 2
hidden_size = 8

learning_rate = 0.01
batch_size = 32

amount = -1      # Tamanho do dataset  |  se negativo, usará o data set todo

''' observação  
O tamanho dos dados do df com o predict é amout
O tamanho dos dados de treino é igual a 3 * EPOCHS
O tamanho dos dados de validação é igual à  EPOCHS
'''


# MODOS
EXPORT_DATA = True    # Exporta arquivos .csv para analizar o resultado da rede
GRAPHS = False        # Mostrar os gráficos
SAVE = False           # Salvar o modelo
# 0 para treinar a rede              | 1 para apenas rodar a rede em uma serie de dados
MODE = 1
GPU = 0               # 0 para uso da CPU                  | 1 para uso da GPU
# 0 para TREINO COM O DF FAKE(HOT)   | 1 para TREINO COM O DF REAL (SONIC)
REAL = 1


# Carregamento dos dados para dentro da rede
if (REAL):
    # Carregar a rede com os dados reais
    df = pd.read_csv(
        '/home/lucasdu/algoritmo/cluster_architecture/sonic_df.csv', sep=",")
    print("\n\nTreino com DataFrame dos dados do sensor sônico\n\n")
    input_df_name = "voltage"
    output_df_name = "velocity_sonic"
else:
    # Carregar a rede com os dados sintéticos
    df = pd.read_csv(
        './data_df.csv', sep=",")
    print("\n\nTreino com DataFrame dos dados sintéticos (gerados através da função sintética)\n\n")

    input_df_name = "voltage"
    output_df_name = "velocity_hotfilm"

# Seleção do dispositivo de processamento
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if GPU == False:
    device = 'cpu'
print("\tDevice de processamento:", device)

# Cabeçalho
print(
    f' -- -- Tipos dos dados do df-- --  \n----------------------------\n{df.dtypes}\n----------------------------')
if amount > 0:
    df = df.head(amount)
amount = len(df)
print(
    f"\n\t __ - __ A quantidade de dados do TS a serem processados pela rede é: {amount} __ - __\n")
print(df)


'''
    Definição da classe que controla os parâmetros da arquitetura da rede 
        - Número de camadas e neurônios de cada rede
        - formato de entrada e saída da rede 
        - definição das funções de ativação
    
'''


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=hidden_size, num_hidden_layers=hidden_layers):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x


'''
    Uso de uma classe para criar os objetos dataset para treino 
        - Separados por dataset de treino e de validação
'''


class VoltageVelocityDataset(Dataset):
    def __init__(self, data):
        self.X = (torch.tensor(
            data[input_df_name].values).float().unsqueeze(1)).to(device)
        self.Y = (torch.tensor(
            data[output_df_name].values).float().unsqueeze(1)).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def export_data(df, predictions, see_train_loss, see_val_loss):
    predictions = pd.DataFrame(predictions)
    see_train_loss = pd.DataFrame(see_train_loss)
    see_val_loss = pd.DataFrame(see_val_loss)

    df_exp = df[['time', 'voltage', 'velocity_sonic']]
    df_exp = df_exp.join(predictions)
    df_exp = df_exp.rename(columns={df_exp.columns[-1]: 'predicts'})

    caminho_completo = os.path.join('/home/lucasdu/algoritmo/cluster_architecture', f'resultado_{SERIE}')
    if not os.path.exists(caminho_completo):
        os.makedirs(caminho_completo)

    df_exp.to_csv(
        f'{caminho_completo}/resultado_predict_{SERIE}.csv', index=False)
    see_train_loss.to_csv(
        f'{caminho_completo}/resultado_train_{SERIE}.csv', index=False)
    see_val_loss.to_csv(
        f'{caminho_completo}/resultado_val_{SERIE}.csv', index=False)
    
    diff_media, diff_max, diff_min = trained_info(df, predictions)
    temp_df = {
        'Média': diff_media,
        'Máximo': diff_max,
        'Mínimo': diff_min
    }
    df_dif = pd.DataFrame(
        {
            'Média': [],
            'Máximo': [],
            'Mínimo': [],
        }
    )
    df_dif.loc[0] = temp_df
    print('\ndf_dif\n',df_dif)
    df_dif.to_csv(f'{caminho_completo}/resultado_diferences_{SERIE}.csv', index=False)
    
    print(f'\nMédia da diferença: {diff_media:6.6f}\nMáxima diferença:   {diff_max:6.6f}\nMínima diferença:   {diff_min:6.6f}\n')


def show_graphs(data, predictions, see_train_loss, see_val_loss):
    # Showing data
    shown = predictions
    shown = shown.assign(original=data[output_df_name])
    see_train_loss = pd.DataFrame(see_train_loss)
    see_val_loss = pd.DataFrame(see_val_loss)
    see_train_loss = see_train_loss.drop(0)
    see_val_loss = see_val_loss.drop(0)
    # pd.set_option('display.max_rows', None)
    print(shown)

    plt.figure(0)
    # Plotting both the curves simultaneously
    if (REAL):
        plt.plot(data.time, data.velocity_sonic, color='r', label='data')
    else:
        plt.plot(data.time, data.velocity_hotfilm, color='r', label='data')
    plt.plot(data.time, predictions,
             color='g', label='processed')
    plt.xlabel("time")
    plt.ylabel("Velocity")
    plt.title("Comparação da velocidade provida da rede e do dataset")
    plt.legend()

    plt.figure(2)
    plt.title("Evolução do erro de treino ao longo do tempo")
    plt.plot(see_train_loss[see_train_loss.columns[0]], see_train_loss[see_train_loss.columns[1]],
             color='g', label='train')
    plt.xlabel("Interação")
    plt.ylabel("Erro")
    plt.legend()

    plt.figure(3)
    plt.title("Evolução do erro da validação ao longo do tempo")
    plt.plot(see_val_loss[see_val_loss.columns[0]], see_val_loss[see_val_loss.columns[1]],
             color='r', label='validation')
    plt.xlabel("Interação")
    plt.ylabel("Erro")
    plt.legend()

    # To load the display window
    plt.show()


def train(data):
    # Split the data into training and validation sets
    train_data = data.sample(frac=0.8, random_state=42)
    val_data = data.drop(train_data.index)

    # Create datasets and dataloaders for training and validation
    train_dataset = VoltageVelocityDataset(train_data)
    val_dataset = VoltageVelocityDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)
    # print(f"\n\n train_dataset\n{train_dataset.X}\nY\n{train_dataset.Y}")

    # Define the MLP and loss function
    mlp = MLP(input_dim=1, output_dim=1, hidden_dim=hidden_size,
              num_hidden_layers=hidden_layers)
    mlp = mlp.to(device)

    criterion = nn.MSELoss()

    # Define the optimizer and learning rate
    optimizer = optim.Adam(mlp.parameters(), lr=0.01)

    # Train the MLP on the entire dataset
    see_train_loss = np.empty([1, 2]).astype(float)
    see_val_loss = np.empty([1, 2]).astype(float)
    idx_train = 0
    idx_val = 0

    for epoch in range(EPOCHS):
        train_loss = 0.0
        for X, Y in train_loader:
            # Forward pass
            outputs = mlp(X)
            loss = criterion(outputs, Y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            see_train_loss = np.append(
                see_train_loss, [idx_train, loss.item()]).reshape(-1, 2)
            train_loss += loss.item() * X.shape[0]
            idx_train = idx_train+1

        # Compute validation loss

        with torch.no_grad():
            val_loss = 0.0
            for X, Y in val_loader:
                outputs = mlp(X)
                loss = criterion(outputs, Y)
                see_val_loss = np.append(
                    see_val_loss, [idx_val, loss.item()]).reshape(-1, 2)
                val_loss += loss.item() * X.shape[0]
                idx_val = idx_val+1

        # Print progress
        if epoch % 100 == 0:
            print("| Epoch {:4} | train loss {:4.4f} | val loss {:4.4f} | ".format(
                epoch, train_loss / len(train_dataset), val_loss / len(val_dataset)))
    print('\n\nidx_train, idx_val: \n', idx_train, idx_val)
    return mlp, see_train_loss, see_val_loss


# Evaluate the MLP on the entire dataset
def predict(mlp, data):
    with torch.no_grad():
        X = torch.tensor(
            data[input_df_name].values).float().unsqueeze(1).to(device)
        Y = torch.tensor(
            data[output_df_name].values).float().unsqueeze(1).to(device)
        predictions = mlp(X)
        accuracy = ((predictions - Y) ** 2).mean().sqrt().item()
        print("Test accuracy| Mean Loss: {:.4f}".format(accuracy))
        return predictions, accuracy


def trained_info(data, predicted):
    df_data = data[output_df_name]
    diff = (predicted - df_data)
    diff_media = diff.abs().mean().to_numpy()[0]
    diff_max = diff.abs().max().to_numpy()[0]
    diff_min = diff.abs().min().to_numpy()[0]
    # print(f"df_data:\n{df_data}, predicted:\n{predicted}")
    # print( f'\nMédia da diferença: {diff_media:6.6f}\nMáxima diferença:   {diff_max:6.6f}\nMínima diferença:   {diff_min:6.6f}\n')
    return diff_media, diff_max, diff_min


def save_model(model):
    torch.save(model.state_dict(
    ), '/home/lucasdu/algoritmo/cluster_architecture/model_mlp_v6.0_cluster.pth')


def main():
    model, train_loss, validation_loss = train(df)
    predicted, accuracy = predict(model, df)
    predicted = pd.DataFrame(predicted)
    train_loss = pd.DataFrame(train_loss)
    validation_loss = pd.DataFrame(validation_loss)

    train_loss = train_loss.rename(columns={train_loss.columns[0]: 'time'})
    train_loss = train_loss.rename(
        columns={train_loss.columns[1]: 'error_train'})
    validation_loss = validation_loss.rename(
        columns={validation_loss.columns[0]: 'time'})
    validation_loss = validation_loss.rename(
        columns={validation_loss.columns[1]: 'error_val'})
    trained_info(df, predicted)

    if SAVE == True:
        save_model(model)

    # print("Treino\n", pd.DataFrame(np.round(train_loss, 3)).head(50))
    # print("Validation\n", pd.DataFrame(np.round(validation_loss, 3)).head(50))

    if EXPORT_DATA == True:
        export_data(df, predicted, train_loss, validation_loss)
    # por ultimo, mostrar os graficos
    if GRAPHS == True:
        show_graphs(df, predicted, train_loss, validation_loss)


if __name__ == '__main__':
    main()
