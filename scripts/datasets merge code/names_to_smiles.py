import pubchempy as pcp

filename = "uniqueDart"
file = open(filename + ".txt", "r")
lines = file.readlines()
smileslist = []
inchikeylist = []
for l in lines:

    if l == "null\n":
        smileslist.append("null")
    else:
        c = pcp.get_compounds(l.split("\n")[0], 'name')

        if c == []:
            smileslist.append("null")
            inchikeylist.append("null")
        else:
            smileslist.append(c[0].isomeric_smiles)
            inchikeylist.append(c[0].inchikey)

print(smileslist)
fw1 = open(filename + "_smiles.txt", "w+")
fw1.writelines("\t\n".join(smileslist))
fw1.close()

fw2 = open(filename + "_inchikey.txt", "w+")
fw2.writelines("\t\n".join(inchikeylist))
fw2.close()

file.close()