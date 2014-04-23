create_proj_instance = """
class createProjInstance 
{
public:
    createProjInstance() {};
    
    /**
     *    @brief      instantiate a projection object or returns previous exsisting one.
     *    @details    called by cpp method ANNarchy::ANNarchy() or by 
     *                createProjInstance::getInstanceOf(int, int, int, int, int)
     */
    Dendrite* getInstanceOf(int ID, Population *pre, Population *post, int postNeuronRank, int target, bool rateCoded) {
        
        if(pre == NULL || post == NULL) {
            std::cout << "Critical error: invalid pointer in c++ core library." << std::endl;
            std::string tmp = rateCoded ? "true" : "false";
            std::cout << "pre = "<< pre <<", post = " << post << ", rateCoded = "<< tmp << std::endl;
            return NULL;
        }
        
        // search for already existing instance
        Dendrite* dendrite = post->getDendrite(postNeuronRank, target, pre);
        
        if(dendrite)
        {
            // return existing one
            return dendrite;
        }
        else
        {
            switch(ID) 
            {
%(case1)s
                default:
                {
                    std::cout << "Unknown typeID: "<< ID << std::endl;
                    return NULL;
                }
            }                    
        }
    }

    /**
     *  @brief          instantiate a projection object or returns previous exsisting one.
     *  @details        called by cython wrapper.
     */
    Dendrite* getInstanceOf(int ID, int preID, int postID, int postNeuronRank, int target, bool rateCoded) 
    {
        Population *pre  = Network::instance()->getPopulation(preID, rateCoded);
        Population *post = Network::instance()->getPopulation(postID, rateCoded);
        
        return getInstanceOf(ID, pre, post, postNeuronRank, target, rateCoded);
    }

};
"""

#
#    Create a complete list of pyx modules
#
py_extension = """include "Network.pyx"

%(pop_inc)s  

include "Dendrite.pyx"
%(proj_inc)s  

%(profile)s
include "Connector.pyx"
"""