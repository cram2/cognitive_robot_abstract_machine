# Segmind

A python library for segmenting simulation episodes of activities. This is done by detecting physical interactions,
and events in the simulation.
This library also integrates with [NEEMPycramInterface](https://github.com/AbdelrhmanBassiouny/NEEMPyCRAMInterface) to segment NEEM episodes as a use case.
## Installation

```bash
pip install segmind
```

## Example Usage

All below examples assume the neems are located in a 'test' database at 'localhost' which can be accessed by 'newuser'
using password 'password'.

### Replaying the motions of a NEEM and segmenting it:

This is done by using the PyCRAMNEEMInterface class which provides an easy way to replay the motions of a NEEM,
then using the NEEMSegmenter class which implements EpisodeSegmenter to segment the motions into activities, actions,
and events.

```Python
from neem_pycram_interface import PyCRAMNEEMInterface
from segmind.neem_segmenter import NEEMSegmenter

from pycram.datastructures.enums import WorldMode
from pycram.worlds.bullet_world import BulletWorld

BulletWorld(WorldMode.GUI)
pni = PyCRAMNEEMInterface('mysql+pymysql://newuser:password@localhost/test')
ns = NEEMSegmenter(pni, annotate_events=True)
ns.start([15])
```

https://github.com/AbdelrhmanBassiouny/EpisodeSegmenter/assets/36744004/186bfd79-f30b-4b4f-ae84-03d2bcce821a


https://github.com/AbdelrhmanBassiouny/EpisodeSegmenter/assets/36744004/bb2e7c65-93cc-4f83-becb-1cb06fcf73ad

