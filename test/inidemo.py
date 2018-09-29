import ConfigParser

config = ConfigParser.ConfigParser()
config.read('config.ini')
secs = config.sections()
print( secs, type(secs))

print( config.get('PATH','src_image_path'))
config.set('PATH','src_image_path','dflakjerouasdfklj')
print( config.get('PATH','src_image_path'))

#with open('config.ini', 'wb') as configfile:
#    config.write(configfile)