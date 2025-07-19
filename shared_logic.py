# shared_logic.py
import cv2
import numpy as np
import random
import logging

# --- CONFIGURAÇÃO DE LOGGING ---
log = logging.getLogger(__name__)

def _is_mask_valid(mask_np, img_shape, img_gray, filter_params):
    """
    Função auxiliar que aplica todos os filtros a uma única máscara.
    Retorna True se a máscara for um balão válido, False caso contrário.
    """
    img_h, img_w = img_shape[:2]
    img_area = img_h * img_w

    # 1. Filtro de Área
    mask_area = np.sum(mask_np)
    if not (
        filter_params["min_area_ratio"] * img_area
        < mask_area
        < filter_params["max_area_ratio"] * img_area
    ):
        return False

    # 2. Filtro de Bounding Box e Proporção
    x, y, w, h = cv2.boundingRect(mask_np)
    if h == 0:
        return False
    aspect_ratio = w / h
    if not (filter_params["min_aspect_ratio"] < aspect_ratio < filter_params["max_aspect_ratio"]):
        return False

    # 3. Filtro de Margem da Borda
    margin = filter_params["border_margin"]
    if (
        x < margin
        or y < margin
        or (x + w) > (img_w - margin)
        or (y + h) > (img_h - margin)
    ):
        return False

    # 4. Filtro de Cor Média (brilho)
    avg_color = cv2.mean(img_gray, mask=mask_np)[0]
    if avg_color < filter_params["min_avg_color"]:
        return False

    # 5. Filtro de Solidez
    contours, _ = cv2.findContours(mask_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(contour)
    if contour_area == 0:
        return False

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False

    solidity = float(contour_area) / hull_area
    if solidity < filter_params["min_solidity"]:
        return False

    return True

def find_speech_balloons(sam_masks, original_image, filter_params):
    """
    Filtra as máscaras geradas pelo SAM para encontrar apenas balões de diálogo.
    """
    if not sam_masks:
        return []

    img_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    filtered_balloons = []
    for mask in sam_masks:
        mask_np = mask.data[0].cpu().numpy().astype(np.uint8)
        
        if _is_mask_valid(mask_np, original_image.shape, img_gray, filter_params):
            filtered_balloons.append(mask_np)
            
    return filtered_balloons

def process_batch(sam_model, image_bytes_list, worker_config, confidence_list):
    """Processa um lote de imagens."""
    results = []
    images = [cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR) for img_bytes in image_bytes_list]
    
    for i, image_original in enumerate(images):
        if image_original is None:
            log.error(f"Falha ao decodificar a imagem no índice {i}.")
            results.append((None, None))
            continue

        confidence = confidence_list[i]
        sam_results = sam_model(image_original, stream=False, verbose=False, conf=confidence)
        
        masks = sam_results[0].masks if sam_results and sam_results[0].masks else []
        detected_balloons = find_speech_balloons(masks, image_original, worker_config["BALLOON_FILTER_PARAMS"])
        log.info(f"Encontrados {len(detected_balloons)} balões válidos na imagem {i}.")

        diag_image = image_original.copy()
        for mask in detected_balloons:
            color = [random.randint(50, 200), random.randint(50, 255), random.randint(50, 200)]
            diag_image[mask == 1] = color
        
        alpha = 0.6
        final_diag_image = cv2.addWeighted(diag_image, alpha, image_original, 1 - alpha, 0)
        
        _, buffer_img = cv2.imencode('.jpg', final_diag_image)
        result_image_bytes = buffer_img.tobytes()

        annotation_text = ""
        if worker_config["GENERATE_YOLO_ANNOTATIONS"] and detected_balloons:
            img_h, img_w, _ = image_original.shape
            annotation_lines = []
            for mask in detected_balloons:
                x, y, w, h = cv2.boundingRect(mask)
                x_center_norm = (x + w / 2) / img_w
                y_center_norm = (y + h / 2) / img_h
                width_norm = w / img_w
                height_norm = h / img_h
                annotation_lines.append(
                    f"{worker_config['YOLO_CLASS_ID']} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
                )
            annotation_text = "\n".join(annotation_lines)

        results.append((result_image_bytes, annotation_text))

    return results