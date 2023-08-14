import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIR = sys.argv[1]
SERIE = sys.argv[2]

input_df_name = "voltage"
output_df_name = "velocity_sonic"


# Generalizando código para ser usado em mais de uma máquina
dir_base = f'/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/neural_network_2023/pytorch_in/src/cluster_architecture/Dados_gerados/{DIR}'


def show_graphs(data, predictions, see_train_loss, see_val_loss):
    see_train_loss = see_train_loss.drop(0)
    see_val_loss = see_val_loss.drop(0)

    # Showing data

    shown = predictions
    print('\n\n\n\nPREVISTO\n\n\n\n', predictions)
    shown = shown.assign(original=data[output_df_name])
    # pd.set_option('display.max_rows', None)
    print("shown\n")
    print(shown)

    plt.figure(0)
    # Plotting both the curves simultaneously
    if (output_df_name == "velocity_sonic"):
        plt.plot(predictions.time, predictions.velocity_sonic,
                 color='r', label='data')
    elif (output_df_name == "velocity_hotfilm"):
        plt.plot(data.time, data.velocity_hotfilm, color='r', label='data')

    plt.plot(predictions.time, predictions.predicts,
             color='g', label='processed')
    plt.xlabel("time")
    plt.ylabel("Velocity")
    plt.title("Comparação da velocidade provida da rede e do dataset")
    plt.legend()

    plt.figure(1)
    plt.plot(predictions.time, predictions.predicts, color='g', label='data')
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("Espéctro da linha temporal")
    plt.legend()

    # plt.figure(2)
    # plt.title("Evolução do erro de treino ao longo do tempo")
    # print('see_train_loss\n',see_train_loss)
    # plt.plot(see_train_loss['time'], see_train_loss['error_train'],color='g', label='train')
    # plt.xlabel("Interação")
    # plt.ylabel("Erro")
    # plt.legend()

    # plt.figure(3)
    # plt.title("Evolução do erro da validação ao longo do tempo")
    # plt.plot(see_val_loss['time'], see_val_loss['error_val'],color='r', label='validation')
    # plt.xlabel("Interação")
    # plt.ylabel("Erro")
    # plt.legend()

    # To load the display window
    plt.show()


print('\n\n\t dir e série: || ', DIR, SERIE, '\n')

# ler todos os dados
data = pd.read_csv('/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/neural_network_2023/pytorch_in/src/cluster_architecture/sonic_df.csv', sep=',')
predict = pd.read_csv(
    f'/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/neural_network_2023/pytorch_in/src/cluster_architecture/Dados_gerados/{DIR}/resultado_{SERIE}/resultado_predict_{SERIE}.csv', sep=',')
train = pd.read_csv(
    f'/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/neural_network_2023/pytorch_in/src/cluster_architecture/Dados_gerados/{DIR}/resultado_{SERIE}/resultado_train_{SERIE}.csv', sep=',')
val = pd.read_csv(f'/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/neural_network_2023/pytorch_in/src/cluster_architecture/Dados_gerados/{DIR}/resultado_{SERIE}/resultado_val_{SERIE}.csv', sep=',')
print(data)

show_graphs(data.head(len(predict)), predict, train, val)
