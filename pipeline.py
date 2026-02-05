# ======================== PIPELINE ========================
import os
import re
import json
import time
import cv2
import numpy as np
from glob import glob
from PIL import Image
import rarfile

import torch
import torch.nn as nn
from torchvision import transforms, models

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.utils import register_keras_serializable
from skimage import measure
from scipy import stats
import open_clip

# ======================== CONFIG ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_if_needed(archive_file, target_folder):
    if os.path.exists(target_folder):
        return
    if not os.path.exists(archive_file):
        return
    print(f"Extracting {archive_file} ...")
    with rarfile.RarFile(archive_file) as r:
        r.extractall(".")
    print("Extraction finished.")

# Auto-unzip data if running on Streamlit/GitHub
extract_if_needed("Project_Data.rar", "Project Data")

# Food/Fruit classifier
FOOD_FRUIT_MODEL_PATH = "Models/part_a_best_mobilenet.pth"
IMG_SIZE_FF = 224

# Fruit classifier
FRUIT_MODEL_PATH = "Models/MobileNetV2_PartC.keras"
TRAIN_DIR_FRUIT = "Project Data/Fruit/Train"
IMG_SIZE_FRUIT = 350

# Food directories for CLIP representatives
TRAIN_DIR_FOOD = "Project Data/Food/Train"
VALID_DIR_FOOD = "Project Data/Food/Validation"

# Calories
FOOD_CALORIES_FILE_TRAIN = "Project Data/Food/Train Calories.txt"
FOOD_CALORIES_FILE_VALID = "Project Data/Food/Val Calories.txt"
FRUIT_CALORIES_FILE = "Project Data/Fruit/Calories.txt"

# Binary segmentation
SEG_MODEL_PATH = "Models/segnet_best.keras"
SEG_IMAGE_SIZE = (256, 256)

# Multi-class segmentation
MULTI_SEG_MODEL_PATH = "Models/best_multiclass_resnet50_unet.keras"
CLASS_MAPPING_FILE = "Models/class_mapping.json"
COLOR_MAPPING_FILE = "Models/color_mapping.json"
MULTI_SEG_IMG_SIZE = 224

# CLIP model
FINETUNED_MODEL_PATH = "Models/clip_finetuned_5_Shots.pth"

# ======================== FOOD/FRUIT CLASSIFIER ========================
class FoodFruitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = models.mobilenet_v2(weights=None)
        self.mobilenet.classifier[1] = nn.Linear(
            self.mobilenet.classifier[1].in_features, 2
        )

    def forward(self, x):
        return self.mobilenet(x)

# Load Food/Fruit model
food_fruit_model = FoodFruitClassifier()
checkpoint = torch.load(FOOD_FRUIT_MODEL_PATH, map_location=DEVICE)
food_fruit_model.load_state_dict(checkpoint["model_state_dict"])
food_fruit_model.to(DEVICE)
food_fruit_model.eval()

transform_ff = transforms.Compose([
    transforms.Resize((IMG_SIZE_FF, IMG_SIZE_FF)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_food_or_fruit(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform_ff(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = food_fruit_model(img)
        pred = torch.argmax(out, dim=1).item()
    return "Food" if pred == 0 else "Fruit"

# ======================== FRUIT CLASSIFICATION ========================
fruit_model = tf.keras.models.load_model(FRUIT_MODEL_PATH)
fruit_classes = sorted([d for d in os.listdir(TRAIN_DIR_FRUIT) if os.path.isdir(os.path.join(TRAIN_DIR_FRUIT, d))])

def preprocess_fruit(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE_FRUIT, IMG_SIZE_FRUIT))
    img = image.img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

def recognize_fruit(img_path):
    pred = fruit_model.predict(preprocess_fruit(img_path), verbose=0)
    return fruit_classes[np.argmax(pred)]

# ======================== CALORIES ========================
def load_calories(file_path):
    calories = {}
    with open(file_path) as f:
        for line in f:
            m = re.match(r"(.+?):\s*~?([\d.]+)", line.strip())
            if m:
                cls_name = m.group(1).strip().lower().replace(" ", "_")
                calories[cls_name] = float(m.group(2))
    return calories

food_calories = {}
food_calories.update(load_calories(FOOD_CALORIES_FILE_TRAIN))
food_calories.update(load_calories(FOOD_CALORIES_FILE_VALID))
fruit_calories = load_calories(FRUIT_CALORIES_FILE)

def extract_grams(name):
    m = re.search(r"(\d+)g", name)
    return int(m.group(1)) if m else 100  # default 100g

# ======================== BINARY SEGMENTATION ========================
seg_model = tf.keras.models.load_model(SEG_MODEL_PATH, compile=False)

def run_binary_segmentation(img_path, save_dir):
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None: return
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, SEG_IMAGE_SIZE)
    img_input = np.expand_dims(img_resized.astype("float32")/255.0, axis=0)
    mask = np.squeeze(seg_model.predict(img_input, verbose=0))
    mask_binary = (mask>0.5).astype(np.uint8)
    mask_resized = cv2.resize(mask_binary, (w,h), interpolation=cv2.INTER_NEAREST)
    mask_final = mask_resized * 255
    save_path = os.path.join(save_dir, os.path.splitext(filename)[0]+"_mask.png")
    cv2.imwrite(save_path, mask_final)
    return save_path

# ======================== MULTI-CLASS SEGMENTATION ========================
with open(CLASS_MAPPING_FILE) as f: class_mapping = json.load(f)
with open(COLOR_MAPPING_FILE) as f: color_mapping = json.load(f)
reverse_mapping = {v:k for k,v in class_mapping.items()}

@register_keras_serializable()
class MultiClassDiceCoefficient(tf.keras.metrics.Metric):
    def __init__(self, num_classes=31, name='dice_coefficient', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_dice = self.add_weight(name='total_dice', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        dice_sum = 0.0
        valid_classes = 0.0
        for i in range(self.num_classes):
            class_idx = tf.cast(i, tf.float32)
            y_t = tf.cast(tf.equal(y_true, class_idx), tf.float32)
            y_p = tf.cast(tf.equal(y_pred, class_idx), tf.float32)
            inter = tf.reduce_sum(y_t * y_p)
            union = tf.reduce_sum(y_t) + tf.reduce_sum(y_p)
            present = tf.cast(tf.reduce_sum(y_t)>0, tf.float32)
            dice = tf.math.divide_no_nan(2.0*inter, union)
            dice_sum += dice*present
            valid_classes += present
        batch_dice = tf.math.divide_no_nan(dice_sum, valid_classes)
        self.total_dice.assign_add(batch_dice)
        self.count.assign_add(1.0)
    def result(self): return tf.math.divide_no_nan(self.total_dice, self.count)
    def reset_state(self):
        self.total_dice.assign(0.0)
        self.count.assign(0.0)
    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

@register_keras_serializable()
def combined_loss(y_true, y_pred):
    ce_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))
    return ce_loss

def dice_loss(y_true, y_pred): return tf.reduce_mean(y_pred)

multi_seg_model = tf.keras.models.load_model(
    MULTI_SEG_MODEL_PATH,
    custom_objects={"combined_loss": combined_loss,"dice_loss": dice_loss,"MultiClassDiceCoefficient": MultiClassDiceCoefficient},
    compile=False
)

def preprocess_image(img_rgb, img_size):
    img = tf.convert_to_tensor(img_rgb)
    img = tf.image.resize(img, (img_size, img_size))
    img = preprocess_input(img)
    img = tf.expand_dims(img, axis=0)
    return img

def clean_mask_advanced(pred_mask):
    if hasattr(pred_mask, 'numpy'): pred_mask = pred_mask.numpy()
    pred_mask = np.array(pred_mask)
    binary_mask = (pred_mask>0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    labels = measure.label(binary_mask)
    cleaned_mask = pred_mask.copy()
    for region in measure.regionprops(labels):
        if region.area<50:
            coords = region.coords
            cleaned_mask[coords[:,0], coords[:,1]] = 0
            continue
        coords = region.coords
        region_vals = pred_mask[coords[:,0], coords[:,1]]
        region_vals = region_vals[region_vals>0]
        if len(region_vals)>0:
            most_frequent_class = stats.mode(region_vals, keepdims=True)[0][0]
            cleaned_mask[coords[:,0], coords[:,1]] = most_frequent_class
    return cleaned_mask

def run_multiclass_segmentation(img_path, save_dir):
    img = cv2.imread(img_path)
    if img is None: return
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = preprocess_image(img_rgb, MULTI_SEG_IMG_SIZE)
    pred = multi_seg_model.predict(inp, verbose=0)[0]
    raw_mask = np.argmax(pred, axis=-1)
    mask_clean = clean_mask_advanced(raw_mask)

    detected_indices = np.unique(mask_clean)
    detected_indices = detected_indices[detected_indices != 0]

    colored = np.zeros((MULTI_SEG_IMG_SIZE, MULTI_SEG_IMG_SIZE, 3), dtype=np.uint8)
    class_names_detected = []
    for idx in detected_indices:
        cls_name = reverse_mapping.get(int(idx), "background")
        rgb_color = np.array(color_mapping.get(cls_name,[0,0,0]),dtype=np.uint8)
        bgr_color = rgb_color[::-1]
        colored[mask_clean==idx] = bgr_color
        class_names_detected.append(cls_name)

    colored = cv2.resize(colored,(w,h),interpolation=cv2.INTER_NEAREST)
    header_height = 30 + 20*len(class_names_detected)
    result_img = np.zeros((h+header_height,w,3),dtype=np.uint8)
    result_img[:header_height,:,:]=0
    for i, cls_name in enumerate(class_names_detected):
        cv2.putText(result_img, cls_name, (10,25+i*20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    result_img[header_height:,:,:] = colored
    save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(img_path))[0]+"_multiseg_mask.png")
    cv2.imwrite(save_path, result_img)
    return save_path

# ======================== CLIP FOOD RECOGNITION ========================
clip_model, _, preprocess_clip = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="openai"
)
checkpoint = torch.load(FINETUNED_MODEL_PATH, map_location=DEVICE)
clip_model.load_state_dict(checkpoint["clip_model_state_dict"])
clip_model.to(DEVICE).eval()

def get_clip_embedding(img_path):
    img = preprocess_clip(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = clip_model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb

print("Loading food representatives for CLIP recognition...")
def load_food_representatives_clip(dirs=[TRAIN_DIR_FOOD, VALID_DIR_FOOD], samples_per_class=5):
    reps = {}
    for d in dirs:
        if not os.path.exists(d): continue
        for cls in os.listdir(d):
            cls_path = os.path.join(d, cls)
            if not os.path.isdir(cls_path) or cls in reps: continue
            imgs = glob(os.path.join(cls_path, "*.jpg"))[:samples_per_class]
            if imgs:
                reps[cls] = [get_clip_embedding(img) for img in imgs]
    return reps

food_reps = load_food_representatives_clip()
print(f"✓ Loaded {len(food_reps)} food classes for CLIP")

def recognize_food_clip(img_path):
    query_emb = get_clip_embedding(img_path)
    best_cls = None
    best_score = -1
    for cls, emb_list in food_reps.items():
        for emb in emb_list:
            score = (query_emb @ emb.T).item()
            if score > best_score:
                best_score = score
                best_cls = cls
    return best_cls

# ======================== FULL PIPELINE ========================
def predict_image(img_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    result = {}

    # Step 1: Food vs Fruit
    main_class = predict_food_or_fruit(img_path)
    result['main_class'] = main_class

    # Step 2: Sub-class recognition
    if main_class == "Food":
        sub_class = recognize_food_clip(img_path)
        cal = food_calories.get(sub_class.lower().replace(" ","_"),0)
    else:
        sub_class = recognize_fruit(img_path)
        cal = fruit_calories.get(sub_class.lower().replace(" ","_"),0)
    result['sub_class'] = sub_class

    # Step 3: Compute calories
    grams = extract_grams(os.path.basename(img_path))
    result['grams'] = grams
    result['total_calories'] = grams*cal

    # Step 4: Segmentation (for fruits)
    if main_class == "Fruit":
        binary_mask_path = run_binary_segmentation(img_path, output_dir)
        multi_mask_path = run_multiclass_segmentation(img_path, output_dir)
        result['binary_mask'] = binary_mask_path
        result['multi_mask'] = multi_mask_path
    else:
        result['binary_mask'] = None
        result['multi_mask'] = None

    # Save result.txt
    with open(os.path.join(output_dir,"result.txt"),"w") as f:
        f.write(f"{main_class}\n{sub_class}\n{result['total_calories']:.2f}\n")

    return result





