import os
import numpy as np
from ase.units import *
from ase.calculators.calculator import Calculator
import torch
import torch.autograd.functional as F
import yaml

from newtonnet.layers.activations import get_activation_by_string
from newtonnet.models import NewtonNet
from newtonnet.data import ExtensiveEnvironment
from newtonnet.data import batch_dataset_converter


##-------------------------------------
##     ML model ASE interface
##--------------------------------------
class MLAseCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'hessian']

    ### Constructor ###
    def __init__(self, model_path=None, settings_path=None, hess_method=None, hess_precision=None, disagreement='std', device='cpu', dtype='float', h=1e-3, **kwargs):
        """
        Constructor for MLAseCalculator

        Parameters
        ----------
        model_path: str or list of str
            path to the model. eg. '5k/models/best_model_state.tar'
        settings_path: str or list of str
            path to the .yml setting path. eg. '5k/run_scripts/config_h2.yml'
        hess_method: str
            method to calculate hessians. 
            None: do not calculate hessian (default)
            'autograd': automatic differentiation
            'fwd_diff': forward difference
            'cnt_diff': central difference
        hess_precision: float
            hessian gradient calculation precision for 'fwd_diff' and 'cnt_diff', ignored otherwise (default: None)
        disagreement: str
            method to calculate disagreement between models.
            'std': standard deviation of all votes (default)
            'std_outlierremoval': standard deviation with outlier removal
            'range': range of all votes
            'values': values of each vote
            None: do not calculate disagreement
        device: 
            device to run model. eg. 'cpu', ['cuda:0', 'cuda:1']
        kwargs
        """
        Calculator.__init__(self, **kwargs)

        if type(device) is list:
            self.device = [torch.device(item) for item in device]
        else:
            self.device = [torch.device(device)]

        self.hess_method = hess_method
        if self.hess_method == 'autograd':
            self.return_hessian = True
        elif self.hess_method == 'fwd_diff' or self.hess_method == 'cnt_diff':
            raise NotImplementedError
            self.return_hessian = False
            self.hess_precision = hess_precision
        else:
            self.return_hessian = False

        self.disagreement = disagreement

        # torch.set_default_tensor_type(torch.DoubleTensor)
        if dtype == 'float':
            self.dtype = torch.float
        elif dtype == 'double':
            self.dtype = torch.double
        if model_path == None:
            self.models = [self.load_model(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model_state.tar"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")
            )]
            print("model_path is None, using default model path and settings path for Transition1x dataset")
        elif type(model_path) is list:
            self.models = [self.load_model(model_path_, settings_path_) for model_path_, settings_path_ in zip(model_path, settings_path)]
        else:
            self.models = [self.load_model(model_path, settings_path)]

        self.ZEROTH_ORDER_H = h
    
    def add_perturbation(self, data, perturbation):
        print('='*89)
        print('len data', len(data))
    
        print(data)
        print('shape of position', data.pos.shape, type(data.pos))
        mean,std = 0.0,0.1 # mean of the Gaussian noise
        noise = torch.normal(mean=torch.full(data.pos.shape, mean), 
                            std=torch.full(data.pos.shape, std))

        # Add the noise to the original tensor
        data_perturb = data.clone()
        data_perturb.pos = data.pos + noise
        print(data)
        print(data_perturb)

    def calculate(self, atoms=None, properties=['energy','forces','hessian'], system_changes=None, random_perturb=False):
        super().calculate(atoms,properties,system_changes)
        if random_perturb:
            data, _p = self.data_formatter(atoms, random_perturb)
        else: 
            data = self.data_formatter(atoms)
        data_ = self.extensive_data_loader(data=data, device=self.device[0])
        pred = self.models[0](data_)
        self.results['energy'] = pred['E'].squeeze(0).detach().cpu().numpy() * (kcal/mol)
        if "forces" in properties:
            self.results['forces'] = pred['F'].squeeze(0).detach().cpu().numpy() * (kcal/mol/Ang)
        if "hessian" in properties:
            self.results['hessian'] = pred['H'].squeeze(0).detach().cpu().numpy() * (kcal/mol/Ang/Ang)
        if random_perturb:
            self.results['perturb'] = _p
        
        del pred

    def load_model(self, model_path, settings_path):
        settings = yaml.safe_load(open(settings_path, "r"))
        activation = get_activation_by_string(settings['model']['activation'])
        model = NewtonNet(resolution=settings['model']['resolution'],
                            n_features=settings['model']['n_features'],
                            activation=activation,
                            n_interactions=settings['model']['n_interactions'],
                            dropout=settings['training']['dropout'],
                            max_z=10,
                            cutoff=settings['data']['cutoff'],  ## data cutoff
                            cutoff_network=settings['model']['cutoff_network'],
                            normalize_atomic=settings['model']['normalize_atomic'],
                            requires_dr=settings['model']['requires_dr'],
                            device=self.device[0],
                            create_graph=False,
                            shared_interactions=settings['model']['shared_interactions'],
                            return_hessian=self.return_hessian,
                            double_update_latent=settings['model']['double_update_latent'],
                            layer_norm=settings['model']['layer_norm'],
                            )

        model.load_state_dict(torch.load(model_path, map_location=self.device[0])['model_state_dict'], )
        model = model
        model.to(self.device[0])
        model.to(self.dtype)
        model.eval()
        return model
    

    def data_formatter(self, atoms, random_perturb = False):
        """
        convert ase.Atoms to input format of the model

        Parameters
        ----------
        atoms: ase.Atoms

        Returns
        -------
        data: dict
            dictionary of arrays with following keys:
                - 'R':positions
                - 'Z':atomic_numbers
                - 'E':energy
                - 'F':forces
        """
        data  = {
            'R': np.array(atoms.get_positions())[np.newaxis, ...], #shape(ndata,natoms,3)
            'Z': np.array(atoms.get_atomic_numbers())[np.newaxis, ...], #shape(ndata,natoms)
            'E': np.zeros((1,1)), #shape(ndata,1)
            'F': np.zeros((1,len(atoms.get_atomic_numbers()), 3)),#shape(ndata,natoms,3)
        }
        if random_perturb:
            _p = np.random.randn(*data['R'].shape) * self.ZEROTH_ORDER_H
            data['R'] += _p
            return data, _p
        else: 
            return data

    def extensive_data_loader(self, data, device=None):
        batch = {'R': data['R'], 'Z': data['Z']}
        N, NM, AM, _, _ = ExtensiveEnvironment().get_environment(data['R'], data['Z'])
        batch.update({'N': N, 'NM': NM, 'AM': AM})
        batch = batch_dataset_converter(batch, device=device)
        batch['R'] = batch['R'].to(self.dtype)
        return batch
    

    def remove_outlier(self, data, idx):
        if idx is None:
            return data
        else:
            return np.delete(data, idx, axis=0)


    def q_test(self, data):
        """
        Dixon's Q test for outlier detection

        Parameters
        ----------
        data: 1d array with shape (nlearners,)

        Returns
        -------
        idx: int or None
            the index to be filtered out (return only one index for now as the default is only 4 learners)
        """
        if len(data) < 3:
            idx = None
        else:
            q_ref = { 3: 0.970,  4: 0.829,  5: 0.710, 
                      6: 0.625,  7: 0.568,  8: 0.526,  9: 0.493, 10: 0.466, 
                     11: 0.444, 12: 0.426, 13: 0.410, 14: 0.396, 15: 0.384, 
                     16: 0.374, 17: 0.365, 18: 0.356, 19: 0.349, 20: 0.342, 
                     21: 0.337, 22: 0.331, 23: 0.326, 24: 0.321, 25: 0.317, 
                     26: 0.312, 27: 0.308, 28: 0.305, 29: 0.301, 30: 0.290}.get(len(self.models))  # 95% confidence interval
            sorted_data = np.sort(data, axis=0)
            q_stat_min = (sorted_data[1] - sorted_data[0]) / (sorted_data[-1] - sorted_data[0])
            q_stat_max = (sorted_data[-1] - sorted_data[-2]) / (sorted_data[-1] - sorted_data[0])
            if q_stat_min > q_ref:
                idx = np.argmin(data)
            elif q_stat_max > q_ref:
                idx = np.argmax(data)
            else:
                idx = None
        return idx


    def calculate_finite_difference_element_wise(self, j_atom, j_idx, add_shift: bool, atoms=None, properties=['energy', 'forces', 'stress'], system_changes=None):
        super().calculate(atoms, properties, system_changes)
        data = self.data_formatter(atoms)
        data["R"][j_atom, j_idx] += self.ZEROTH_ORDER_H / 2 if add_shift else - self.ZEROTH_ORDER_H / 2
        data_ = self.extensive_data_loader(data=data, device=self.device[0])
        pred = self.models[0](data_)
        self.results['energy'] = pred['E'].squeeze(0).detach().cpu().numpy() * (kcal/mol)
        if "forces" in properties:
            self.results['forces'] = pred['F'].squeeze(0).detach().cpu().numpy() * (kcal/mol/Ang)
        if "hessian" in properties:
            self.results['hessian'] = pred['H'].squeeze(0).detach().cpu().numpy() * (kcal/mol/Ang/Ang)
        
        del pred
