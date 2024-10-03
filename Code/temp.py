import pickle
with open('Data/UserGoal.pkl','rb') as handle:
    goal = pickle.load(handle)
print(len(goal))
print(goal)
