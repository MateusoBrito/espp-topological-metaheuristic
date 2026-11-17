#include <stdio.h>
#include <stdlib.h>
#include <math.h> // Necessário para a raiz quadrada

int main() {
    FILE *file = fopen("data/processed/distance_matrix.bin", "rb");
    if (!file) {
        printf("Erro ao abrir o arquivo binario.\n");
        return 1;
    }

    // 1. Descobre o tamanho do arquivo em bytes
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file); // Volta o ponteiro para o início do arquivo

    // 2. Calcula o N automaticamente
    // file_size = N * N * sizeof(double)
    int N = (int) sqrt(file_size / sizeof(double));
    printf("Carregando matriz de distancias %d x %d (Tamanho do arquivo: %ld bytes)\n", N, N, file_size);

    double *matriz_distancias = (double *)malloc(N * N * sizeof(double));
    if (!matriz_distancias) {
        printf("Erro de alocacao de memoria.\n");
        return 1;
    }

    // 3. Lê a matriz inteira (como não tem cabeçalho, lemos desde o byte 0)
    fread(matriz_distancias, sizeof(double), N * N, file);
    fclose(file);

    // --- EXEMPLO ---
    int i = 0; 
    int j = 5;
    double distancia = matriz_distancias[i * N + j];
    printf("A distancia de roteamento entre %d e %d eh: %f km\n", i, j, distancia);

    free(matriz_distancias);
    return 0;
}