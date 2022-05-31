#
# emotion_detection_test_predictions.py
#
# To support Emotion Avatar, output the test CSV with answers provided
# by the model.

from emotion_detection_harness import *

_model_variants_location = "models"
_test_csv_name = "test_annotated.csv"
_max_seq_length = 256
_dataloader_batch_size = 32

def generate_test_csv(model, tokenizer, device, test_set):
  # Prepare the Test Dataloader. 
  sentences_test = test_set["text"]
  input_ids_test, attention_masks_test = train_model_encode_data(tokenizer, sentences_test)
  solutions_test = test_set["solution"].astype(int)
  features_test = (input_ids_test, attention_masks_test, solutions_test)
  features_test_tensors = [torch.tensor(feature, dtype=torch.long) for feature in features_test]
  dataset_test = TensorDataset(*features_test_tensors)
  sampler_test = SequentialSampler(dataset_test)
  dataloader_test = DataLoader(dataset_test, sampler=sampler_test, batch_size=_dataloader_batch_size)

  # Evaluate the model. 
  test_accuracy = 0
  # Set the model in evaluation mode. 
  model.eval()
  total_predictions = []
  for batch in tqdm(dataloader_test, desc="Final Test Eval", total=len(dataloader_test)):
    # For each batch of samples, predict the outputs. 
    input_ids = batch[0].to(device)
    attention_masks = batch[1].to(device)
    labels = batch[2]
    with torch.no_grad():
      outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)

    # Given the outputs, calculate the accuracy. 
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    predictions = np.argmax(logits, axis=1).flatten()
    total_predictions += predictions.tolist()
    labels = labels.numpy().flatten()
    # Accuracy is # correct / # labels. 
    test_accuracy += np.sum(predictions == labels) / len(labels)
  
  # Once all done iterating through all of the dev data,
  # Calculate the total dev acc and keep it for our records. 
  test_accuracy = test_accuracy / len(dataloader_test)

  # Each epoch, output the results. 
  print("[INFO] Model Test Accuracy: " + str(test_accuracy)+ "\n")

  assert len(total_predictions) == len(sentences_test)

  test_set["prediction"] = total_predictions

  print("[INFO] Writing to CSV...")
  test_set.to_csv(_test_csv_name)
  print("[INFO] Done! Goodnight.")

def train_model_encode_data(tokenizer, sentences):
  print("[DEBUG] Encoding a set of " + str(len(sentences)) + " sentences with RoBERTa Large's preprocessing tokenizer.")
  input_ids = []
  attention_masks = []
  # For each sentence, generate an input_id and an attention_mask.
  for sentence in sentences:
    encoded_data = tokenizer.encode_plus(
      sentence,
      max_length=_max_seq_length, 
      padding = "max_length",
      truncation_strategy="longest_first",
      truncation=True)
    input_ids.append(encoded_data["input_ids"])
    attention_masks.append(encoded_data["attention_mask"])
  print("[DEBUG] Encoding complete: generated " + str(len(attention_masks)) + " attention masks.")
  return np.array(input_ids), np.array(attention_masks)

# Given a model_num, return the tokenizer and model stored at the
# expected location. Loads the device to run it on if it is not 
# provided. Also returns the device in case it is needed.
def load_tokenizer_and_model(model_num, device = None, use_cpu = False):
  # Grab the device first if we don't have it. 
  if device is None:
    device = train_model_load_device(use_cpu = use_cpu)

  try:
    model_path = _model_variants_location + "/" + str(model_num)
    print("[DEBUG] Loading Tokenizer for model " + str(model_num) + " from '" + model_path + "'.")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("[DEBUG] Loading Model for model " + str(model_num) + " from '" + model_path + "'.")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    print("[DEBUG] Loading Model onto device.")
    model.to(device)

    return model, tokenizer, device
  except Exception as e:
    print("[ERROR] Unable to load model " + str(model_num) + ". Exception: ")
    print(e)
    return None, None, None

# Load the device for torch work. Expects a boolean indicating whether
# we'll be using the CPU. Returns None in the event of a GPU CUDA load
# failure.
def train_model_load_device(use_cpu):
  device = None

  if not use_cpu:
    # Note we expect to be using CUDA for this training session. If we
    # encounter an error, we'll just stop. 
    try:
      print("[DEBUG] Verifying CUDA: ", end="")
      print(torch.zeros(1).cuda())
      print("[DEBUG] CUDA version: ", end="")
      print(torch.version.cuda)
      print("[DEBUG] Torch CUDA is available: " + str(torch.cuda.is_available()))
    except:
      print("[ERROR] Unable to access Torch CUDA - was pytorch installed from pytorch.org with the correct version?")
      return None
    
    device = torch.device("cuda") 
    print("[DEBUG] GPU with CUDA successfully added as device.")
  else:
    device = torch.device("cpu") # Use the CPU for better debugging messages. 
    print("[DEBUG] CPU successfully added as device.")
  
  return device

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("variant_num")
  parser.add_argument("model_num")
  args = parser.parse_args()

  variant_num = args.variant_num
  model_num = args.model_num

  emotion_detection = EmotionDetectionHarness()
  train_set, dev_set, test_set = emotion_detection.load_train_dev_test(variant_num = variant_num)
  model, tokenizer, device = load_tokenizer_and_model(model_num=model_num)
  generate_test_csv(model, tokenizer, device, test_set)