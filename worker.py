# worker.py
import requests
import time
import base64
import sys
import os
import logging
import uuid
import threading
from tqdm import tqdm
from ultralytics import SAM

from shared_logic import process_batch

# Configuração do logging para o worker
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - (Worker) - %(message)s"
)

# --- ESTADO GLOBAL DO WORKER ---
WORKER_VERSION = "1.0.0"
worker_id = str(uuid.uuid4())
worker_status = "Inicializando"
worker_config = {}

def send_heartbeat(master_url):
    """
    Envia um status (heartbeat) para o master periodicamente em segundo plano.
    """
    global worker_status
    while True:
        try:
            payload = {"worker_id": worker_id, "status": worker_status}
            requests.post(f"{master_url}/worker/heartbeat", json=payload, timeout=10)
        except requests.exceptions.RequestException as e:
            logging.warning(f"Não foi possível enviar heartbeat ao master: {e}")
        time.sleep(15)

def download_file(url, file_path):
    """
    Baixa um arquivo (como o modelo de IA) com uma barra de progresso.
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            desc=f"Baixando {os.path.basename(file_path)}",
        ) as pbar:
            with open(file_path, "wb") as f:
                for data in response.iter_content(chunk_size=1024):
                    pbar.update(len(data))
                    f.write(data)
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Falha no download: {e}")
        return False

def initialize_model(model_name, model_url):
    """
    Verifica se o modelo SAM existe, baixa se necessário, e o carrega na memória.
    """
    if not os.path.exists(model_name):
        logging.info(f"Modelo '{model_name}' não encontrado. Baixando...")
        if not download_file(model_url, model_name):
            return None
    try:
        logging.info("Carregando modelo SAM na memória...")
        model = SAM(model_name)
        logging.info("Modelo carregado com sucesso.")
        return model
    except Exception as e:
        logging.critical(f"Erro fatal ao carregar o modelo SAM: {e}", exc_info=True)
        return None

def get_master_config(master_url):
    """
    Obtém a configuração do master.
    """
    try:
        response = requests.get(f"{master_url}/config", timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Não foi possível obter a configuração do master: {e}")
        return None

def start_worker(master_url):
    """
    Função principal que gerencia o ciclo de vida do worker.
    """
    global worker_status, worker_config
    logging.info(f">>> INICIANDO WORKER (ID: {worker_id}) <<<")

    worker_config = get_master_config(master_url)
    if not worker_config:
        worker_status = "Falha (Config)"
        return

    heartbeat_thread = threading.Thread(
        target=send_heartbeat, args=(master_url,), daemon=True
    )
    heartbeat_thread.start()

    sam_model = initialize_model(worker_config['SAM_MODEL_NAME'], worker_config['SAM_MODEL_URL'])
    if not sam_model:
        worker_status = "Falha (Modelo)"
        return

    while True:
        try:
            worker_status = "Ocioso"
            logging.info("Solicitando novo lote de tarefas ao master...")

            get_jobs_url = f"{master_url}/get_jobs?worker_id={worker_id}&version={WORKER_VERSION}"
            response = requests.get(get_jobs_url, timeout=60)

            if response.status_code == 426: # Upgrade Required
                logging.error("Versão do worker desatualizada. Por favor, atualize o worker e reinicie.")
                worker_status = "Versão Desatualizada"
                break

            response.raise_for_status()
            batch_job = response.json()

            status = batch_job.get("status")
            if status == "new_batch":
                tasks = batch_job["tasks"]
                worker_status = f"Processando {len(tasks)} tarefas"
                logging.info(f"Lote de {len(tasks)} tarefas recebido.")

                image_bytes_list = [base64.b64decode(task["image_data_b64"]) for task in tasks]
                confidence_list = [task.get("confidence") for task in tasks]

                start_time = time.time()
                results = process_batch(sam_model, image_bytes_list, worker_config, confidence_list)
                logging.info(
                    f"Processamento do lote concluído em {time.time() - start_time:.2f}s."
                )

                payloads = []
                for i, (result_bytes, annotation_text) in enumerate(results):
                    if result_bytes:
                        payloads.append({
                            "task_id": tasks[i]["task_id"],
                            "manga_name": tasks[i]["manga_name"],
                            "filename": tasks[i]["filename"],
                            "image_data_b64": base64.b64encode(result_bytes).decode(
                                "utf-8"
                            ),
                            "txt_data": annotation_text,
                        })
                    else:
                        logging.error(
                            f"Erro ao processar a imagem da tarefa {tasks[i]['task_id']}. O master irá tratar."
                        )
                
                if payloads:
                    logging.info(f"Enviando resultados de {len(payloads)} tarefas...")
                    requests.post(
                        f"{master_url}/submit_results", json={"results": payloads}, timeout=30
                    )

            elif status == "no_more_jobs":
                worker_status = "Concluído"
                logging.info("Não há mais tarefas na fila. Encerrando worker.")
                break
            elif status == "paused":
                worker_status = "Aguardando (Master Pausado)"
                logging.info("O master está pausado. Aguardando...")
                time.sleep(worker_config['WORKER_RETRY_DELAY'])
            else:
                time.sleep(5)

        except requests.exceptions.RequestException as e:
            worker_status = "Erro de Conexão"
            logging.error(
                f"Erro de conexão com o master: {e}. Tentando novamente em {worker_config.get('WORKER_RETRY_DELAY', 15)}s."
            )
            time.sleep(worker_config.get('WORKER_RETRY_DELAY', 15))
        except Exception as e:
            worker_status = "Erro Crítico"
            logging.critical(f"Erro inesperado no worker: {e}", exc_info=True)
            time.sleep(worker_config.get('WORKER_RETRY_DELAY', 15))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Uso: python {sys.argv[0]} <master_url>")
        sys.exit(1)

    master_url_arg = sys.argv[1]
    start_worker(master_url_arg)