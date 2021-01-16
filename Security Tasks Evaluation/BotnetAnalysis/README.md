##Dependencies and Data


### Botnets

- Download the P2P and botnet datasets gathered for PeerRush, available [here](http://peerrush.cs.uga.edu/peerrush/)
  - Place them inside `BotnetAnalysis/Data`
- Botnet detection code by Pratik Narang is available [here](https://github.com/pratiknarang/peershark)

### Parse Original Captures Used in PeerShark

- For each dataset (Waledac, Storm, P2P)
  - Run `peershark/FilterPackets.py`
  - Retrieve original parse of the .pcap at `pcapdata` folder

*Note: Storm data samples must be appended with ".pcap"*
`for f in *; do mv "$f" "$f.pcap"; done`
  
### Run the FlowLens botnet detection experiment

Run `fullRun.sh`, which is responsible for applying different quantization parameter combinations on the PL and IPT of P2P packet flows