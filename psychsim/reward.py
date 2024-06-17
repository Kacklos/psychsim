from psychsim.pwl import *
        
def maximizeFeature(key,agent):
    return KeyedTree(setToFeatureMatrix(rewardKey(agent),key,1.))
        
def minimizeFeature(key,agent):
    return KeyedTree(setToFeatureMatrix(rewardKey(agent),key,-1.))


def achieveFeatureValue(key,value,agent):
    return achieve_feature_value(key, value, agent)


def achieve_feature_value(key,value,agent):
    return makeTree({'if': equalRow(key,value),
                     True: setToConstantMatrix(rewardKey(agent),1.),
                     False: setToConstantMatrix(rewardKey(agent),0.)})


def achieveGoal(key, agent, invert=False):
    return achieve_goal(key, agent, invert)


def achieve_goal(key, agent, invert=False):
    """
    :param invert: if C{True}, then reward is gained if the feature is False, 
                   not True (as is the default)
    """
    return makeTree({'if': falseRow(key) if invert else trueRow(key),
                     True: setToConstantMatrix(rewardKey(agent),1.),
                     False: setToConstantMatrix(rewardKey(agent),0.),})

def minimizeDifference(key1,key2,agent):
    return makeTree({'if': greaterThanRow(key1, key2),
                     True: dynamicsMatrix(rewardKey(agent),{key1: -1.,key2: 1.}),
                     False: dynamicsMatrix(rewardKey(agent),{key1: 1.,key2: -1.})})

def null_reward(agent):
    """
    :return: a reward function that always returns 0
    :type agent: str
    """
    return makeTree(setToConstantMatrix(rewardKey(agent), 0))