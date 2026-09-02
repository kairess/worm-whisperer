# 참고 자료

인용 연도와 저자는 계획 단계의 기억에 기반한다. Phase 3에서 프로토콜을 코드화하기 전에 각 항목의 DOI를 확인해 채운다.

## 도구 / 코드

| 이름 | 용도 | 위치 |
|------|------|------|
| c302 | 커넥톰 → NeuroML2 네트워크 생성, 파라미터 세트 A–D | github.com/openworm/c302 |
| pyNeuroML / jNeuroML | NeuroML 실행, NEURON 내보내기 | github.com/NeuroML/pyNeuroML |
| NEURON | 신경 시뮬레이션 백엔드 (`pip install neuron`) | neuron.yale.edu |
| Sibernetic | 3D SPH 신체/유체 시뮬레이션 (OpenCL) | github.com/openworm/sibernetic |
| openworm/openworm (Docker) | c302 + Sibernetic 통합 파이프라인 | hub.docker.com/r/openworm/openworm |
| owmeta (구 PyOpenWorm) | 뉴런/시냅스 메타데이터 질의 | github.com/openworm/owmeta |
| WormAtlas, WormBase | 뉴런 명명, 해부 참고 | wormatlas.org |
| wormsim (Boyle-Berri-Cohen) | 2D 신경역학 이동 모델의 참조 구현 | Boyle et al. 2012 |

## 커넥톰 데이터

- White, Southgate, Thomson, Brenner 1986. *The structure of the nervous system of the nematode C. elegans*. Phil. Trans. R. Soc. B.
- Varshney et al. 2011. *Structural properties of the C. elegans neuronal network*. PLoS Comput Biol.
- Cook et al. 2019. *Whole-animal connectomes of both C. elegans sexes*. Nature. (c302 기본 리더)
- Witvliet et al. 2021. *Connectomes across development reveal principles of brain maturation*. Nature.
- Bentley et al. 2016. *The multilayer connectome of C. elegans*. PLoS Comput Biol. (모노아민/펩타이드 층, 확장 모듈 근거)

## 이동 회로

- Chalfie et al. 1985. *The neural circuit for touch sensitivity in C. elegans*. J Neurosci. (AVA/AVB/PVC/AVD, 터치 회로, 레이저 절제)
- Guo et al. 2009. *Optical interrogation of neural circuits in C. elegans*. Nat Methods. (AVA 광활성화)
- Wen et al. 2012. *Proprioceptive coupling within motor neurons drives C. elegans forward locomotion*. Neuron.
- Fouad et al. 2018. *Distributed rhythm generators underlie C. elegans forward locomotion*. eLife.
- Xu et al. 2018. *Descending pathway facilitates undulatory wave propagation in C. elegans through gap junctions*. PNAS.

## 회전 / 항법

- Gray, Hill, Bargmann 2005. *A circuit for navigation in C. elegans*. PNAS. (SMD, RIV, AIY/AIZ, 국소 탐색)
- Pierce-Shimomura, Morse, Lockery 1999. *The fundamental role of pirouettes in C. elegans chemotaxis*. J Neurosci.
- Iino & Yoshida 2009. *Parallel use of two behavioral mechanisms for chemotaxis in C. elegans*. J Neurosci. (klinotaxis)
- Hendricks et al. 2012. *Compartmentalized calcium dynamics in a C. elegans interneuron encode head movement*. Nature. (RIA)
- Pirri et al. 2009. *A tyramine-gated chloride channel coordinates distinct parts of the escape response*. Neuron.
- Hills, Brockie, Maricq 2004. *Dopamine and glutamate control area-restricted search behavior*. J Neurosci.

## 정지 / 수면 / 행동 상태

- Turek, Lewandrowski, Bringmann 2013. *An AP2 transcription factor is required for a sleep-active neuron to induce sleep-like quiescence in C. elegans*. Curr Biol. (RIS)
- Turek et al. 2016. *Sleep-active neuron specification and sleep induction require FLP-11 neuropeptides*. eLife.
- Hill et al. 2014. *Cellular stress induces a protective sleep-like state in C. elegans*. Curr Biol. (ALA)
- Nelson et al. 2014. *FMRFamide-like FLP-13 neuropeptides promote quiescence following heat stress*. Curr Biol.
- Flavell et al. 2013. *Serotonin and the neuropeptide PDF initiate and extend opposite behavioral states*. Cell. (roaming/dwelling)
- Iwanir et al. 2013. *The microarchitecture of C. elegans behavior during lethargus*. Sleep.

## 신체 역학

- Boyle, Berri, Cohen 2012. *Gait modulation in C. elegans: an integrated neuromechanical model*. Front Comput Neurosci.
- Izquierdo & Beer 2018. *From head to tail: a neuromechanical model of forward locomotion in C. elegans*. Phil Trans R Soc B.
- Pierce-Shimomura et al. 2008. *Genetic analysis of crawling and swimming locomotory patterns in C. elegans*. PNAS.
- Gray & Lissmann 1964. *The locomotion of nematodes*. J Exp Biol. (저항력 이론)
- Palyanov, Khayrulin, Larson 2016/2018. Sibernetic 논문들.

## OpenWorm 전반

- Sarma et al. 2018. *OpenWorm: overview and recent advances in integrative biological simulation of C. elegans*. Phil Trans R Soc B.
- Gleeson et al. 2018. *c302: a multiscale framework for modelling the nervous system of C. elegans*. Phil Trans R Soc B.
