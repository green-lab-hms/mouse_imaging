function printMazeName(sessionFile)

user = getenv("USER");
path = ['/home/' user '/code/ViRMEn_jg'];
addpath(genpath(path));
load(sessionFile)
disp(experData.name)
