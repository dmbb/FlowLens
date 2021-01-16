# FlowLens

This repository holds the code for the paper "FlowLens: Enabling Efficient Flow Classification for ML-based Network Security Applications". 
If you end up using our code for your experiments, please cite our work as follows:

```
@inproceedings{protozoa,
  title={FlowLens: Enabling Efficient Flow Classification for ML-based Network Security Applications},
  author={Barradas, Diogo and Santos, Nuno and Rodrigues, Lu{\'i}s and Signorello, Salvatore and Ramos, Fernando M. V. and Madeira, Andr{\'e}},
  booktitle={Proceedings of the 28th Network and Distributed System Security Symposium},
  year={2021},
  address={San Diego, CA, USA},
}
```

##Dependencies and Data


### General Dependencies

- Install WEKA
- Run `pip install -r requirements.txt`

### Datasets

- Please check the `README.md` in each specific security task folder


## How may I use your code?

- The `Security Tasks Evaluation` folder includes the code we used for evaluating different ML-based security tasks when using FlowLens. The code applies different combinations of our quantization and truncation approaches and allows for checking FlowLens flow markers trade-offs between accuracy and memory footprint

- The `Flow Marker Accumulator` folder includes an adaptation of the P4<sub>16</sub> code we used for implementing FlowLens' flow marker accumulator in a Barefoot Tofino switch. Due to NDA concerns, we make public this adapted version of our code that can be run on the P4's BMV2 behavioral model.


*Todo: Provide a full end-to-end dummy example of FlowLens running in BMV2 - e.g. on P4's tutorial VM.*