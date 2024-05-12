#include "TwoD_evolution_design_chatzip.h"
//#include <iostream>
//#include "TwoD_evolution_design_chatzip.cpp"
#include <ap_fixed.h>
#include <hls_vector.h>
#include <stdint.h>

int main(){
		int steps;
		const int totalIterations=120;
		//int *pu=u;
		//int *pv=v;
		//int *pc=c;
		//int *punew=u_new;
		//int *pvnew=v_new;
		//int *pcnew=c_new;

//		int mid=50;
//		int k=5;
		bool load=1;

/*		int narrow_iterations_limit1 = datat(1.0/4)*totalIterations;
			int narrow_iterations_limit2_tips = datat(1.3/4)*totalIterations;
			int narrow_iterations_limit3 = datat(2.3/4)*totalIterations;
			int narrow_iterations_limit4_tips = datat(2.6/4)*totalIterations;
			int narrow_iterations_limit5 = datat(3.6/4)*totalIterations;

		hls::vector<bool, totalIterations> code_selector = false;

	    for (int i = 0; i < narrow_iterations_limit1; i++)
	        code_selector[i] = true;
	    for (int i = narrow_iterations_limit2_tips; i <= narrow_iterations_limit3; i++)
	        code_selector[i] = true;
	    for (int i = narrow_iterations_limit4_tips; i <= narrow_iterations_limit5; i++)
	        code_selector[i] = true;

*/

		    for (steps=0;steps<totalIterations;steps++)
		    {
		    	evolution_top(load);
		    	load=0;
		    }

		    return 0;
}
