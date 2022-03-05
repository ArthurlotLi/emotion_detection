#
# emotion_detection_data_utility.py
#
# Utility meant to aid in the creation of .csv files that may
# be utilized by dataset generation/preprocessing classes. 
# Some of the datasets are in different forms that will require
# some elbow grease to get into a standardized form. 
#
# This script should be run once to generate the 7 csv files
# located in the ./raw_data folder. Once the files have been
# generated, unique dataset variants may be extracted utilizing
# emotion_detection_dataset.py.
#
# TXT to CSV processing required:
#  2 WASSA-2017 Emotion Intensities(EmoInt)
#    - 4 txt files (clearnly delimited, independent files)
#  3 Cecilia Ovesdotter Alm's Affect data
#    - 176 txt files (cleanly delimited, independent files)
#  4 DailyDialog
#    - 4 txt files (each file is one column - line numbers equivalent)
#  5 Emotion Stimulus
#    - 1 txt file (abnormal xml-inspired format)
#
# No TXT to CSV processing necessary:
#  1 ISEAR
#  6 MELD
#  7 SMILE 
#
# All csv files will need to be modified in order to provide the 
# same harmonized column structure. That sturcture will be as 
# follows, simply:
# ["text", "solution"]
#
# Any additional information will simply be dropped. 
#
# Some datasets will need all rows with certian categories
# dropped as they do not fit the targeted emotion model. 
# Our emotion model follows Paul Ekman's with an additional
# neutral category. 
#
# Datasets with categories to drop:
#  1. ISEAR 
#     - Drop guilt + shame
#  5. Emotion Stimulus
#     - Drop shame
#
# Finally, note that all of the solution names should be harmonized
# as well. The datasets that need harmonizing:
#  3 Cecilia
#     - happy->joy | sad -> sadness | angry -> anger
#  4 DailyDialog
#     - happiness->joy
#  7 Smile
#     - happiness->joy

import os
import pandas as pd

class EmotionDetectionDataUtility:
  # All txt files contained within each dataset's respective
  # directory will be assumed to be part of the dataset. 
  wassa_directory = "./datasets/wasa2017"
  cecilia_directory = "./datasets/cecilia"
  daily_dialog_directory = "./datasets/dailydialog"
  emotion_stimulus_directory = "./datasets/emotionstimulus"

  isear_location = "./datasets/isear.csv"
  meld_location = "./datasets/meld.csv"
  smile_location="./datasets/smile.csv"

  csv_output_directory = "./raw_data"

  # Main function that extracts all 7 harmonized dataset files
  # and writes them to the csv_output_directory. 
  def generate_csv_files(self):
    print("[INFO] Executing Emotion Detection Data Utility.")

    # Generate the csv files that need to be created.
    wassa_data = self.generate_wassa()
    cecilia_data = self.generate_cecilia()
    daily_dialog_data = self.generate_daily_dialog()
    emotion_stimulus_data = self.generate_emotion_stimulus()
    isear_data = self.generate_isear()
    meld_data = self.generate_meld()
    smile_data = self.generate_smile()

    self.write_csv(wassa_data, "wassa2017.csv")
    self.write_csv(cecilia_data, "cecilia.csv")
    self.write_csv(daily_dialog_data, "dailydialog.csv")
    self.write_csv(emotion_stimulus_data, "emotionstimulus.csv")
    self.write_csv(isear_data, "isear.csv")
    self.write_csv(meld_data, "meld.csv")
    self.write_csv(smile_data, "smile.csv")

    print("[INFO] Utility complete. Goodnight...")

  # Returns the combined pandas dataframe of all txt files in
  # the wassa 2017 directory. Expect whitespace to be the
  # delimiter. 
  def generate_wassa(self):
    print("[INFO] Generating Wassa csv.")
    wassa_data = None
    wassa_files_data = []

    for filename in os.listdir(self.wassa_directory):
      if filename.endswith("txt"):
        try:
          print("[DEBUG] Wassa - processing file " + str(filename) + ".")
          file_data = pd.read_csv(self.wassa_directory + "/" + filename, delimiter="\t", names = ["id","text","solution","_"])
          file_data = file_data.drop(["id", "_"], axis=1)
          wassa_files_data.append(file_data)
        except Exception as e:
          print("[ERROR] Wassa - Failed to process file " + str(filename) + ". Error:")
          print(e)
    
    if len(wassa_files_data) > 0:
      wassa_data = wassa_files_data[0]
    
    for i in range(1, len(wassa_files_data)):
      wassa_data = pd.concat([wassa_data, wassa_files_data[i]], axis=0)

    return wassa_data

  # The Cecilia dataset contains a lot of files but they're all 
  # standardized. you'll need to handle the sort of dual-answer
  # nature of the solutions, though. You'll also need to encode
  # the values properly - don't forget to map happy, sad, and
  # angry accordingly. 
  def generate_cecilia(self):
    print("[INFO] Generating Cecilia csv.")
    cecilia_data = None
    cecilia_files_data = []

    for filename in os.listdir(self.cecilia_directory):
      if filename.endswith("emmood"):
        try:
          print("[DEBUG] Cecilia - processing file " + str(filename) + ".")
          file_data = pd.read_csv(self.cecilia_directory + "/" + filename, delimiter="\t", names = ["sentid:sentid","emlabelA:emlabelb","moodlabela:moodlabelb","text"])

          # For each file, drop the things we don't need. 
          file_data = file_data.drop(["sentid:sentid", "moodlabela:moodlabelb"], axis=1)

          # There are two labels provided per passage - it is unclear
          # what differences are between them. Use Emotion Label B.
          #file_data["solution"] = file_data["emlabelA:emlabelb"].replace(["H:(.+)","Sa:(.+)","F:(.+)","A:(.+)","D:(.+)","Su[.+-]:(.+)","N:(.+)"], ["joy","sadness","fear","anger","disgust","surprise","neutral"], regex=True)
          file_data["solution"] = file_data["emlabelA:emlabelb"].replace(["(.+):H","(.+):Sa","(.+):F","(.+):A","(.+):D","(.+):Su[.+-]","(.+):N"], ["joy","sadness","fear","anger","disgust","surprise","neutral"], regex=True)
          
          # Rearrange columns. 
          file_data = file_data[["text", "solution"]]

          cecilia_files_data.append(file_data)
        except Exception as e:
          print("[ERROR] Cecilia - Failed to process file " + str(filename) + ". Error:")
          print(e)
    
    if len(cecilia_files_data) > 0:
      cecilia_data = cecilia_files_data[0]
    
    for i in range(1, len(cecilia_files_data)):
      cecilia_data = pd.concat([cecilia_data, cecilia_files_data[i]], axis=0)
    
    return cecilia_data

  # The Daily Dialog dataset will need to be concatenated on axis
  # 1 instead of 0, as each txt file contains the same number of lines
  # but a different column. Don't forget to map happiness->joy. 
  #
  # Another consideration - each line does not have one solution.
  # There is a solution per sentence, and each line can have multiple
  # sentences. 
  def generate_daily_dialog(self):
    print("[INFO] Generating Daily Dialog csv.")
    daily_dialog_data = None
    daily_dialog_files_data = []

    for filename in os.listdir(self.daily_dialog_directory):
      if filename.endswith("txt"):
        try:
          print("[DEBUG] DailyDialog - processing file " + str(filename) + ".")
          column_name = filename.replace("dialogues_", "")
          column_name = column_name.replace(".txt", "")
          file_data = pd.read_csv(self.daily_dialog_directory + "/" + filename, delimiter="\t", names = [column_name])
          daily_dialog_files_data.append(file_data)
        except Exception as e:
          print("[ERROR] DailyDialog - Failed to process file " + str(filename) + ". Error:")
          print(e)
    
    if len(daily_dialog_files_data) > 0:
      daily_dialog_data = daily_dialog_files_data[0]
    
    for i in range(1, len(daily_dialog_files_data)):
      daily_dialog_data = pd.concat([daily_dialog_data, daily_dialog_files_data[i]], axis=1)

    # Now we need to create a new dataframe from the one we've just
    # generated, separating out sentences into their own rows with
    # their own solutions. Each instance is separated by the delimiter
    # __eou__.
    solution_column = []
    text_column = []
    for index, row in daily_dialog_data.iterrows():
      # Ignore the last empty element (there's a __eou__ at the end of all)
      sentences = row["text"].split("__eou__")[:-1] 
      emotions = row["emotion"].split()
      if len(sentences) != len(emotions):
        print("[WARNING] DailyDialog - Encountered an entry that did not have equivalent sentences to emotions. (sentences "+ str(len(sentences)) + " =/= emotions " + str(len(emotions)) + ") Skipping.")
      else:
        # All good, like-indexed. 
        for i in range(0, len(sentences)):
          emotion = emotions[i]
          # Replace emotion codes with actual string values. 
          if emotion == "0": emotion = "neutral"
          elif emotion == "1": emotion = "anger"
          elif emotion == "2": emotion = "disgust"
          elif emotion == "3": emotion = "fear"
          elif emotion == "4": emotion = "joy" # mapping Happiness to Joy. 
          elif emotion == "5": emotion = "sadness"
          elif emotion == "6": emotion = "surprise"
          else: 
            print("[WARNING] DailyDialog - Encountered an emotion " + str(emotion) + " that did not have an equivalent class.")
            continue
          solution_column.append(emotion)
          text_column.append(sentences[i])
    
    # Generate a new dataframe. 
    daily_dialog_data = pd.DataFrame(data={"text":text_column,"solution":solution_column})
    
    return daily_dialog_data

  # Emotion Stimulus has some strange seemingly xml-inspired node
  # format. You'll need to go line by line and create a dataframe of
  # your own. Don't forget to drop the shame category as well. 
  def generate_emotion_stimulus(self):
    print("[INFO] Generating Emotion Stimulus csv.")
    emotion_stimulus_data = None

    for filename in os.listdir(self.emotion_stimulus_directory):
      if filename.endswith("txt"):
        try:
          print("[DEBUG] EmotionStimulus - processing file " + str(filename) + ".")
          file_data = pd.read_csv(self.emotion_stimulus_directory + "/" + filename, delimiter="\n", names = ["nodes"])
          # We only expect one file. 
          emotion_stimulus_data = file_data
          break
        except Exception as e:
          print("[ERROR] EmotionStimulus - Failed to process file " + str(filename) + ". Error:")
          print(e)

    # Now we need to parse each row and extract the emotion from the
    # row node name. Ex) <happy>text</happy>
    solution_column = []
    text_column = []
    for index, row in emotion_stimulus_data.iterrows():
      original_node = row["nodes"]
      split_last_node = original_node.split("<\\")
      emotion = split_last_node[1].replace(">","").strip()
      text = split_last_node[0].split(">")[1].strip()

      # Drop the shame emotion.
      if emotion != "shame":
        # Need to map Happy to Joy and Sad to Sadness.
        if emotion == "happy": emotion = "joy"
        elif emotion == "sad": emotion = "sadness"

        solution_column.append(emotion)
        text_column.append(text)
    
    emotion_stimulus_data = pd.DataFrame(data={"text":text_column, "solution":solution_column})
    
    return emotion_stimulus_data

  # The ISEAR dataset is pretty straightforward - looks like there is
  # a phantom column to get rid of, though. Change the order and drop
  # rows with solutions of guilt or shame. 
  def generate_isear(self):
    print("[INFO] Generating ISEAR csv.")
    isear_data = None
    try:
      print("[DEBUG] ISEAR - processing file " + str(self.isear_location) + ".")
      file_data = pd.read_csv(self.isear_location, delimiter=",", names = ["solution","text", "_"], encoding='utf8')
      isear_data = file_data
    except Exception as e:
      print("[ERROR] ISEAR - Failed to read file " + str(self.isear_location) + ". Error:")
      print(e)
      return

    # There are some typos we need to address. 
    text_column = []
    solution_column = []
    for index, row in isear_data.iterrows():
      emotion = row["solution"]
      text = row["text"]
      # Split the entry if there is a matchup. 
      emotion = emotion.split("|")[0] 
      # Drop classes that we cannot easily fold into other categories. 
      if emotion == "shame" or emotion == "guilt":
        continue
      # Print out any typos that we run into. 
      if emotion != "disgust" and emotion != "surprise" and emotion != "joy" and emotion != "sadness" and emotion != "fear" and emotion != "anger":
        print("[WARNING] ISEAR - encountered an unknown emotion " + str(emotion) + "! Skipping...")
        continue
    
      solution_column.append(emotion)
      text_column.append(text)

    isear_data = pd.DataFrame(data={"text":text_column,"solution":solution_column})
    
    return isear_data

  # The MELD dataset is about as clean as it gets. Kudos to these 
  # folks. 
  def generate_meld(self):
    print("[INFO] Generating MELD csv.")
    try:
      print("[DEBUG] MELD - processing file " + str(self.meld_location) + ".")
      file_data = pd.read_csv(self.meld_location)
      meld_data = file_data
    except Exception as e:
      print("[ERROR] MELD - Failed to read file " + str(self.meld_location) + ". Error:")
      print(e)
      return

    # Rearrange columns. Spring cleaning. 
    meld_data["text"] = meld_data["Utterance"]
    meld_data["solution"] = meld_data["Emotion"]
    meld_data = meld_data[["text","solution"]]
    
    return meld_data

  # The SMILE dataset is rather messy - solutions can contain a | bar 
  # with two emotions, while a large majority have "nocode" solutions. 
  # This is different from instances with "not-relevant" solutions. 
  def generate_smile(self):
    print("[INFO] Generating Smile csv.")
    try:
      print("[DEBUG] SMILE - processing file " + str(self.smile_location) + ".")
      file_data = pd.read_csv(self.smile_location, names = ["_", "text", "orig_solution"])
      smile_data = file_data
    except Exception as e:
      print("[ERROR] SMILE - Failed to read file " + str(self.smile_location) + ". Error:")
      print(e)
      return

    # Address the wide variety of classes. 
    text_column = []
    solution_column = []
    for index, row in smile_data.iterrows():
      emotion = row["orig_solution"]
      text = row["text"]
      # Split the entry if there is a matchup. 
      emotion = emotion.split("|")[0] 
      if emotion == "happy": emotion = "joy"
      elif emotion == "sad": emotion = "sadness"
      elif emotion == "angry": emotion = "anger"
      elif emotion == "nocode": emotion = "neutral"
      elif emotion == "not-relevant":
        continue
      elif emotion != "disgust" and emotion != "surprise":
        print("[WARNING] SMILE - encountered an unknown emotion " + str(emotion) + "! Skipping...")
        continue
    
      solution_column.append(emotion)
      text_column.append(text)

    smile_data = pd.DataFrame(data={"text":text_column,"solution":solution_column})
    
    return smile_data

  # Given a pandas dataframe, output the dataframe to a csv file names
  # appropriately. 
  def write_csv(self, data, filename):
    if data is not None and filename is not None:
      print("[INFO] Writing " + filename + " with data shape of ", end="")
      print(data.shape, end="")
      print(".") 

      data.to_csv(self.csv_output_directory + "/" + filename, encoding='utf-8', index=False)

if __name__ == "__main__":
  data_utils = EmotionDetectionDataUtility()
  data_utils.generate_csv_files()