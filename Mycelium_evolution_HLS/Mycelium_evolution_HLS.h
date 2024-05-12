#ifndef TwoD_evolution_design_chatzip
#define TwoD_evolution_design_chatzip
#include <ap_fixed.h>
#include <stdint.h>

//typedef float datat;
//typedef float datat_gamma;

typedef ap_fixed<16,8,AP_RND_CONV,AP_SAT> datat;
typedef ap_fixed<16,11,AP_RND_CONV,AP_SAT> datat_gamma;
//typedef double datat;

//template<int N, int M>
void evolution(datat u[100][100], datat v[100][100], datat c[100][100], datat u_new[100][100], datat v_new[100][100], datat c_new[100][100],datat ij_mat[100][100],datat n[100][100], bool code_selector);

datat evolution_top(bool load);

#endif
