#include "form.h"
#include "stiffness_operator_basix_table.hpp"
#include "stiffness_operator_ffc_table.hpp"
#include <cmath>
#include <dolfinx.h>

using namespace dolfinx;

int main(int argc, char* argv[]){
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);
  {
	std::cout.precision(18);

	// Create mesh and function space
	std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(mesh::create_box(
	  MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {1, 1, 1},
	  mesh::CellType::hexahedron, mesh::GhostMode::none));

	std::shared_ptr<fem::FunctionSpace> V = std::make_shared<fem::FunctionSpace>(
	  fem::create_functionspace(functionspace_form_form_a, "u", mesh));

	// Get index map and block size
	std::shared_ptr<const common::IndexMap> index_map = V->dofmap()->index_map;
	int bs = V->dofmap()->index_map_bs();

	// Create stiffness operator
	std::shared_ptr<fem::Function<double>> u = std::make_shared<fem::Function<double>>(V);
	// xtl::span<double> _u = u->x()->mutable_array();
	// std::fill(_u.begin(), _u.end(), 100);

    u->interpolate(
        [](auto& x) -> xt::xarray<PetscScalar>
        {
          auto dx = xt::square(xt::row(x, 0) - 0.5)
                    + xt::square(xt::row(x, 1) - 0.5);
          return 10e10 * xt::exp(-(dx) / 0.02);
        });


	std::shared_ptr<StiffnessOperatorFFCX<double>> stiffness_operator_ffcx = std::make_shared<StiffnessOperatorFFCX<double>>(V);
	std::shared_ptr<la::Vector<double>> s_ffcx = std::make_shared<la::Vector<double>>(index_map, bs);
	tcb::span<double> _s_ffcx = s_ffcx->mutable_array();
	std::fill(_s_ffcx.begin(), _s_ffcx.end(), 0.0);
	stiffness_operator_ffcx->operator()(*u->x(), *s_ffcx);


	std::shared_ptr<StiffnessOperatorBASIX<double>> stiffness_operator_basix = std::make_shared<StiffnessOperatorBASIX<double>>(V);
	std::shared_ptr<la::Vector<double>> s_basix = std::make_shared<la::Vector<double>>(index_map, bs);
	tcb::span<double> _s_basix = s_basix->mutable_array();
	std::fill(_s_basix.begin(), _s_basix.end(), 0.0);
	stiffness_operator_basix->operator()(*u->x(), *s_basix);

	for (int i = 0; i < 10; ++i){
    }
	std::cout << std::endl;
	for (int i = 0; i < 10; ++i){
      std::cout << s_basix->mutable_array()[i] << " " << s_ffcx->mutable_array()[i] << " " <<
        (s_basix->mutable_array()[i] - s_ffcx->mutable_array()[i]) / s_ffcx->mutable_array()[i] << std::endl;
    }
  }
}
