
from aids import AID
#i is aid
assay = AID('1346982')
print(assay.get_property('Name')[0])
print('\n')
print(assay.get_property('Description'))
