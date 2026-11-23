#!/usr/bin/env bash
set -e

# ==============================================================================
# CONFIGURAÇÃO DE ESCOPO E PARÂMETROS DO MODELO DE WAXMAN
# ==============================================================================
# Você pode escolher o estado de duas formas:
#   1) Passando na linha de comando:
#        ./scripts/run_graphs_main.sh SE
#        ./scripts/run_graphs_main.sh MG
#        ./scripts/run_graphs_main.sh BR
#      (Opcionalmente passando alpha e beta customizados: ./scripts/run_graphs_main.sh SE 0.25 0.055)
#
#   2) Ou alterando a variável UF_PADRAO abaixo e rodando apenas:
#        ./scripts/run_graphs_main.sh
# ==============================================================================

UF_PADRAO="SE"   # Opções testadas: "BR", "MG", "SE" (ou qualquer outra sigla de UF)

UF="${1:-$UF_PADRAO}"
UF=$(echo "$UF" | tr '[:lower:]' '[:upper:]')

# ------------------------------------------------------------------------------
# TABELA DE CALIBRAÇÃO DE ALPHA E BETA POR ESCALA GEOGRÁFICA (Diâmetro L):
#
# 1) BRASIL TOTAL (UF="BR")
#    - Torres: ~85.564 | Diâmetro L: ~4.499 km
#    - ALPHA = 0.40 | BETA = 0.0003
#    - Alcance característico (beta * L): ~1,35 km | Raio máx busca: ~8,09 km
#
# 2) MINAS GERAIS (UF="MG" - Estado de grande porte)
#    - Torres: ~9.368  | Diâmetro L: ~1.239 km
#    - ALPHA = 0.40 | BETA = 0.015
#    - Alcance característico (beta * L): ~18,59 km | Raio máx busca: ~111,36 km
#
# 3) SERGIPE (UF="SE" - Estado compacto, L ~6x menor que MG)
#    - Torres: ~755    | Diâmetro L: ~213 km
#    - Por ter L pequeno (213 km), usar beta=0.015 deixaria beta*L = 3,2 km,
#      desconectando metade do interior. Com BETA=0.055 e ALPHA=0.25, temos:
#    - Alcance característico (beta * L): ~11,73 km | Raio máx busca: ~70,28 km
#    - Conecta ~99% das torres (747/755) no maior componente sem saturar Aracaju.
# ------------------------------------------------------------------------------

case "$UF" in
    BR)
        DEFAULT_ALPHA="0.4"
        DEFAULT_BETA="0.0003"
        ;;
    MG)
        DEFAULT_ALPHA="0.4"
        DEFAULT_BETA="0.015"
        ;;
    SE)
        DEFAULT_ALPHA="0.25"
        DEFAULT_BETA="0.055"
        ;;
    *)
        # Padrão genérico para demais estados
        DEFAULT_ALPHA="0.4"
        DEFAULT_BETA="0.015"
        ;;
esac

ALPHA="${2:-$DEFAULT_ALPHA}"
BETA="${3:-$DEFAULT_BETA}"

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/graphs_${UF}_${TIMESTAMP}.log"

echo "=================================================================="
echo " Executando pipeline para UF=${UF} | alpha=${ALPHA} | beta=${BETA}"
echo " Log salvo em: ${LOG}"
echo "=================================================================="

PYTHON_BIN="python3"
if [ -x ".venv/bin/python3" ]; then
    PYTHON_BIN=".venv/bin/python3"
fi

"$PYTHON_BIN" -u -m src.main --uf "$UF" --alpha "$ALPHA" --beta "$BETA" | tee -a "$LOG"
