# Detecting Recon Cyberattacks with Machine Learning

## Objective
The objective of this project is to develop a model that can reliably predict the occurrence of Recon Cyberattacks in the context of the CICIOT 2023 dataset. The idea is to specialize models to detect one kind of cyberattack, thus augmenting their performance, as the researchers also pledged:
> “Finally, although we are focusing on 33 different attacks, future directions could also be tailored to address issues related to individual attacks or categories.”
> Neto, E.C.P.; Dadkhah, S.; Ferreira, R.; Zohourian, A.; Lu, R.; Ghorbani, A.A. CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment. Sensors 2023, 23, 5941. [https://doi.org/10.3390/s23135941](https://doi.org/10.3390/s23135941)

## The CICIOT Dataset
This dataset is the result of research from the University of New Brunswick Centre for Cybersecurity. It has extracted CSV features on network traffic across 105 Internet of Things (IoT) devices with 33 cyberattacks run on them. 7 types of attacks were run: distributed denial of service (DDoS), denial of service (DoS), reconnaissance, web-based, brute-force, spoofing, and the Mirai botnet.
[Link to Kaggle](https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset).

## Simplified Data Features Overview

### Communication Patterns
- **Duration Measures**: `flow_duration`, `Duration`
- **Transmission Rates**: `Rate`, `Srate`, `Drate`

### Traffic Signs
- **Flag Counts**: `fin_flag_number`, `syn_flag_number`, `rst_flag_number`, etc.
- **Packet Counts**: `ack_count`, `syn_count`, `fin_count`, etc.

### Communication Types
- **Protocol Indicators**: `HTTP`, `HTTPS`, `DNS`, `Telnet`, etc.

### Conversation Statistics
- **Size Measures**: `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size`
- **Timing Measures**: `IAT` (Inter-Arrival Time)

### Interaction Complexity
- **Complexity Metrics**: `Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight`

### Nature of Traffic
- **Traffic Category**: `label`

## Directory Structure
- Notebooks representing three different stages of the project:
  - `Cybersec_eda.ipynb`
  - `Cybersec_binary_modelling.ipynb`
  - `Cybersec_multiclass_modelling.ipynb`
- `Pipeline_adaboost.zip` contains the binary classification model.
- `Gridsearch_xgboost.pkl` contains the multiclassification model.
- `Deployment.py` contains the final inference file.

## Process Description
1. Data collection and preprocessing.
2. Exploratory Data Analysis (EDA).
3. Experimentation with different pipelines for binary classification.
4. Cross validation and hyperparameter tuning.
5. Development of a two-stage model for multiclass classification.
6. Guidelines for model inference provided in `deployment.py`.


## Feature Descriptions

| Feature           | Data Type | Description                                                  |
|-------------------|-----------|--------------------------------------------------------------|
| `flow_duration`   | float64   | Total duration of the network flow in seconds.               |
| `Header_Length`   | float64   | The length of the packet header in bytes.                    |
| `Protocol Type`   | float64   | Numerical representation of the network protocol used.       |
| `Duration`        | float64   | Time duration of the network connection (similar to flow duration but could be a subset or different measurement). |
| `Rate`            | float64   | The rate of packet transmission over the network in packets per second. |
| `Srate`           | float64   | The rate of outbound packets in the flow, indicating data sent from the source. |
| `Drate`           | float64   | The rate of inbound packets in the flow, indicating data received by the destination. |
| `fin_flag_number` | float64   | Number of packets with the FIN flag set, indicating the end of data communication. |
| `syn_flag_number` | float64   | Number of packets with the SYN flag set, used to initiate a TCP connection. |
| `rst_flag_number` | float64   | Number of packets with the RST flag set, used to reset the connection. |
| `psh_flag_number` | float64   | Number of packets with the PSH flag set, indicating the push function. |
| `ack_flag_number` | float64   | Number of packets with the ACK flag set, used to acknowledge the receipt of packets. |
| `ece_flag_number` | float64   | Number of packets with the ECE flag set, indicating Explicit Congestion Notification Echo. |
| `cwr_flag_number` | float64   | Number of packets with the CWR flag set, used to signal congestion window reduced. |
| `ack_count`       | float64   | The total number of acknowledgment packets within the flow.  |
| `syn_count`       | float64   | The total number of synchronization packets within the flow. |
| `fin_count`       | float64   | The total number of finish packets within the flow.          |
| `urg_count`       | float64   | The total number of urgent packets within the flow.          |
| `rst_count`       | float64   | The total number of reset packets within the flow.           |
| `HTTP`            | float64   | Indicator of HTTP traffic (1 for HTTP traffic, 0 otherwise). |
| `HTTPS`           | float64   | Indicator of HTTPS traffic (1 for HTTPS traffic, 0 otherwise).|
| `DNS`             | float64   | Indicator of DNS traffic (1 for DNS traffic, 0 otherwise).   |
| `Telnet`          | float64   | Indicator of Telnet traffic (1 for Telnet traffic, 0 otherwise). |
| `SMTP`            | float64   | Indicator of SMTP traffic (1 for SMTP traffic, 0 otherwise). |
| `SSH`             | float64   | Indicator of SSH traffic (1 for SSH traffic, 0 otherwise).   |
| `IRC`             | float64   | Indicator of IRC traffic (1 for IRC traffic, 0 otherwise).   |
| `TCP`             | float64   | Indicator of TCP protocol usage (1 for TCP, 0 otherwise).    |
| `UDP`             | float64   | Indicator of UDP protocol usage (1 for UDP, 0 otherwise).    |
| `DHCP`            | float64   | Indicator of DHCP traffic (1 for DHCP traffic, 0 otherwise). |
| `ARP`             | float64   | Indicator of ARP traffic (1 for ARP traffic, 0 otherwise).   |
| `ICMP`            | float64   | Indicator of ICMP traffic (1 for ICMP traffic, 0 otherwise). |
| `IPv`             | float64   | Indicator of IPv4 or IPv6 traffic (1 for IP traffic, 0 otherwise). |
| `LLC`             | float64   | Indicator of LLC traffic (1 for LLC traffic, 0 otherwise).   |
| `Tot sum`         | float64   | The total size of the packets transferred in the flow.       |
| `Min`             | float64   | The minimum size of packets in the flow.                     |
| `Max`             | float64   | The maximum size of packets in the flow.                     |
| `AVG`             | float64   | The average size of packets in the flow.                     |
| `Std`             | float64   | The standard deviation of packet sizes in the flow.          |
| `Tot size`        | float64   | Total size of the flow in bytes.                             |
| `IAT`             | float64   | Inter-Arrival Time of the packets in the flow.               |
| `Number`          | float64   | Total number of packets in the flow.                         |
| `Magnitue`        | float64   | A derived metric indicating the magnitude of the flow.       |
| `Radius`          | float64   | A derived metric indicating the radius of the flow.          |
| `Covariance`      | float64   | The covariance of packet sizes in the flow.                  |
| `Variance`        | float64   | The variance of packet sizes in the flow.                    |
| `Weight`          | float64   | A weight metric related to the flow.                         |
| `label`           | object    | Number of incoming packets × Number of outgoing packets                      |


## Notes
- Time-consuming cells in Sprint 2 notebook are encapsulated with triple quotations.
- Deliberate avoidance of feature selection steps for intuitive inference process.

