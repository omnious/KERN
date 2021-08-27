import shutil
# import tqdm
'''
original = r'/Users/yeskendir/Desktop/influencer_mappings/influencer_json/6a0c0125-2e0a-11ea-90d5-02607f665cd2.json'
target = r'/Users/yeskendir/Desktop/influencer_mappings/nonempty_influencer_json/6a0c0125-2e0a-11ea-90d5-02607f665cd2.json'
shutil.copyfile(original, target)
'''
'''
original_root = '/Users/yeskendir/Desktop/influencer_mappings/influencer_json/'
target_root = '/Users/yeskendir/Desktop/influencer_mappings/nonempty_influencer_json/'

filename = '6a0c0125-2e0a-11ea-90d5-02607f665cd2.json'

original = original_root + filename
target = target_root + filename
shutil.copyfile(original, target)
'''
original_root = '/home/omnious/workspace/yeskendir/Influencer_Export_autotagged/influencer_json/'
target_root = '/home/omnious/workspace/yeskendir/Influencer_Export_autotagged/nonempty_influencer_json/'
file_with_names = "/home/omnious/workspace/yeskendir/Influencer_Export_autotagged/nonempty_influencers.txt"
file = open(file_with_names, 'r')
count = 0
nonempty_filenames = file.readlines()
for filename in nonempty_filenames:
    filename = filename.strip()
    original = original_root + filename
    target = target_root + filename
    try:
        shutil.copyfile(original, target)
    except:
        print("failed to copy ", original)
    count += 1
    if count%100 == 0:
        print(count)
print("count = ", count)