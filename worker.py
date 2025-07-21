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

from shared_logic import process_image

# --- Configuração de logging, estado global, etc. (sem alterações) ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - (Worker) - %(message)s"
)
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
    # (sem alterações)
    ...

def initialize_model(model_name, model_url):
    # (sem alterações)
    ...

def discover_master_url(discovery_url):
    """
    Função para descobrir a URL real do master (ngrok) a partir de um ponto de entrada fixo.
    """
    for attempt in range(5): # Tenta descobrir 5 vezes
        try:
            logging.info(f"Tentando descobrir a URL do Master em '{discovery_url}' (tentativa {attempt + 1}/5)...")
            response = requests.get(f"{discovery_url}/get_ngrok_url", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            master_url = data.get("ngrok_url")
            if master_url:
                logging.info(f"URL do Master descoberta com sucesso: {master_url}")
                return master_url
            else:
                logging.warning("Resposta do servidor de descoberta não continha a URL.")

        except requests.exceptions.RequestException as e:
            logging.error(f"Erro ao contatar o servidor de descoberta: {e}")
        
        logging.info("Aguardando 10 segundos antes de tentar novamente...")
        time.sleep(10)
    
    logging.critical("Não foi possível descobrir a URL do Master após várias tentativas. Encerrando.")
    return None

def get_master_config(master_url):
    """Obtém a configuração do master."""
    try:
        response = requests.get(f"{master_url}/config", timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Não foi possível obter a configuração do master: {e}")
        return None

def start_worker(discovery_url):
    """
    Função principal que gerencia o ciclo de vida do worker.
    """
    global worker_status, worker_config
    logging.info(f">>> INICIANDO WORKER (ID: {worker_id}) <<<")

    # Passo 1: Descobrir a URL real do Master
    master_url = discover_master_url(discovery_url)
    if not master_url:
        return # Encerra se não conseguir descobrir o master

    # Passo 2: Obter a configuração do Master, usando a URL descoberta
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
    
    BATCH_SIZE = worker_config.get("BATCH_SIZE", 1)

    # Loop principal (usa a variável `master_url` descoberta)
    while True:
        try:
            worker_status = "Ocioso"
            logging.info(f"Solicitando novo lote de tarefas (tamanho: {BATCH_SIZE}) ao master em {master_url}...")

            get_job_url = f"{master_url}/get_batch_jobs?worker_id={worker_id}&version={WORKER_VERSION}&size={BATCH_SIZE}"
            response = requests.get(get_job_url, timeout=60)

            if response.status_code == 426:
                logging.error("Versão do worker desatualizada. Por favor, atualize o worker e reinicie.")
                worker_status = "Versão Desatualizada"
                break

            response.raise_for_status()
            response_data = response.json()
            
            job_list = response_data.get("jobs", [])
            status = response_data.get("status")

            if status == "new_batch" and job_list:
                results_to_submit = []
                logging.info(f"Lote de {len(job_list)} tarefas recebido. Iniciando processamento.")

                for job in job_list:
                    task_id = job["task_id"]
                    worker_status = f"Processando: {job['filename']} (Lote)"
                    logging.info(f"Processando tarefa {task_id} do lote: {job['filename']}")

                    image_bytes = base64.b64decode(job["image_data_b64"])
                    confidence = job.get("confidence")

                    start_time = time.time()
                    result_bytes, annotation_text = process_image(sam_model, image_bytes, worker_config, confidence)
                    logging.info(f"Tarefa {task_id} concluída em {time.time() - start_time:.2f}s.")

                    if result_bytes:
                        payload = {
                            "task_id": task_id,
                            "manga_name": job["manga_name"],
                            "filename": job["filename"],
                            "image_data_b64": base64.b64encode(result_bytes).decode("utf-8"),
                            "txt_data": annotation_text,
                        }
                        results_to_submit.append(payload)
                    else:
                        logging.error(f"Erro ao processar a imagem da tarefa {task_id}. O master irá tratar.")
                
                if results_to_submit:
                    submitted_successfully = False
                    for attempt in range(3):
                        try:
                            logging.info(f"Enviando lote de {len(results_to_submit)} resultados (tentativa {attempt + 1}/3)...")
                            submit_response = requests.post(
                                f"{master_url}/submit_batch_results", json={"results": results_to_submit}, timeout=60
                            )
                            submit_response.raise_for_status()
                            submitted_successfully = True
                            logging.info(f"Lote de resultados enviado com sucesso.")
                            break
                        except requests.exceptions.RequestException as e:
                            logging.warning(f"Falha ao enviar lote de resultados: {e}. Tentando novamente em 5s.")
                            time.sleep(5)
                    
                    if not submitted_successfully:
                        logging.error(f"Não foi possível enviar o lote de resultados após 3 tentativas. O master irá tratar.")

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
    # O URL de descoberta agora está fixo no código.
    # Não é mais necessário passá-lo como argumento.
    DISCOVERY_URL = "http://147.185.221.22:40943"
    
    print(f"Uso: python {sys.argv[0]}")
    print(f"O worker tentará se conectar ao servidor de descoberta em: {DISCOVERY_URL}")

    start_worker(DISCOVERY_URL)
