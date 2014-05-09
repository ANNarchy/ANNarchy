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
    Dendrite* getInstanceOf(int ID, Population *pre, Population *post, int postNeuronRank, int target) {
        
        if(pre == NULL || post == NULL) {
            std::cout << "Critical error: invalid pointer in c++ core library." << std::endl;
            return NULL;
        }
        
        // search for already existing instance
        Dendrite* dendrite = post->getProjection(pre, target)->getDendrite(postNeuronRank);
        
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
    #ifdef _DEBUG
        std::string tmp = rateCoded ? "true" : "false";
        std::cout << "getInstanceOf ( pre = "<< preID <<", post = " << postID << ", rateCoded = "<< tmp << " ) called." << std::endl;
    #endif
    
        Population *pre  = Network::instance()->getPopulation(preID);
        Population *post = Network::instance()->getPopulation(postID);
        
        auto dendrite =  getInstanceOf(ID, pre, post, postNeuronRank, target);
        
        post->getProjection(pre, target)->addDendrite(postNeuronRank, dendrite);        
        return dendrite;
    }

};
"""


proj_instance = """
                case %(id)s:
                {
                #ifdef _DEBUG
                    std::cout << "Instantiate name=%(name)s" << std::endl;
                #endif
                    return new %(name)s(pre, post, postNeuronRank, target);
                }
"""

#
#    Create a complete list of pyx modules
#
py_extension = """include "Network.pyx"

%(pop_inc)s  

include "RateProjection.pyx"
include "SpikeProjection.pyx"
include "Dendrite.pyx"
%(proj_inc)s  

%(profile)s
include "Connector.pyx"
"""