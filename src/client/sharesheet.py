"""
Opens the Android share sheet to send text or files to other applications.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from os.path import exists, basename  # @UnresolvedImport

from kivy.logger import Logger
from jnius import autoclass, cast  # @UnresolvedImport
from android import mActivity, api_version  # @UnresolvedImport
from androidstorage4kivy import SharedStorage  # @UnresolvedImport

#------------------------------------------------------------------------------

JString = autoclass('java.lang.String')
Intent  = autoclass('android.content.Intent')
MediaStoreFiles = autoclass('android.provider.MediaStore$Files')
MediaStoreMediaColumns = autoclass('android.provider.MediaStore$MediaColumns')
ContentValues = autoclass('android.content.ContentValues')
ContentUris = autoclass('android.content.ContentUris')
ArrayList = autoclass('java.util.ArrayList')

#------------------------------------------------------------------------------


class ShareSheet():

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.legacy_uri_list = []

    def share_plain_text(self, plain_text, app = None):
        try:
            self._cleanup_legacy_uri_list()
            self.plain_text = plain_text 
            self.send = Intent()
            self.send.setAction(Intent.ACTION_SEND)  
            self.send.setType("text/plain")
            self.send.putExtra(Intent.EXTRA_TEXT, JString(self.plain_text))
            if app:
                self.send.setPackage(app)
                mActivity.startActivity(self.send)
            else:
                self.send1 = Intent.createChooser(self.send,None)
                mActivity.startActivity(self.send1)
        except Exception as e:
            Logger.warning('ShareSheet().share_plain_text()')
            Logger.warning(str(e))

    def share_file(self, shared_file_path, app = None):
        try:
            self._cleanup_legacy_uri_list()
            if shared_file_path == None:
                return
            uri = self._legacy_create_uri(shared_file_path)
            if uri == None:
                return
            cr =  mActivity.getContentResolver()
            self.MIME = cr.getType(uri)
            self.parcelable = cast('android.os.Parcelable', uri)  
            self.send = Intent()
            self.send.setAction(Intent.ACTION_SEND)  
            self.send.setType(self.MIME)
            self.send.putExtra(Intent.EXTRA_STREAM, self.parcelable)
            self.send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            if app:
                self.send.setPackage(app)
                mActivity.startActivity(self.send)
            else:
                self.choose = Intent.createChooser(self.send, 'Open file')
                mActivity.startActivity(self.choose)
        except Exception as e:
            Logger.warning('ShareSheet().share_file()')
            Logger.warning(str(e))

    def view_file(self, shared_file_path):
        try:
            self._cleanup_legacy_uri_list()
            if shared_file_path == None:
                return
            uri = self._legacy_create_uri(shared_file_path)
            if uri == None:
                return
            cr =  mActivity.getContentResolver()
            self.MIME = cr.getType(uri)
            self.send = Intent()
            self.send.setAction(Intent.ACTION_VIEW)
            self.send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            self.send.setDataAndType(uri, self.MIME)
            mActivity.startActivity(self.send)
        except Exception as e:
            Logger.warning('ShareSheet().view_file()')
            Logger.warning(str(e))

    def share_file_list(self, shared_file_paths, app = None):
        try:
            self._cleanup_legacy_uri_list()  
            if shared_file_paths == None or len(shared_file_paths) == 0:
                return
            if len(shared_file_paths) == 1:
                self.share_file(shared_file_paths[0], app)
                return
            uri_list = []
            for shared_file_path in shared_file_paths:
                uri = self._legacy_create_uri(shared_file_path)
                if uri:
                    uri_list.append(uri)
            if len(uri_list) == 0:
                return
            cr =  mActivity.getContentResolver()
            self.MIME = cr.getType(uri_list[0])
            self.send = Intent()
            self.send.setAction(Intent.ACTION_SEND_MULTIPLE)
            self.send.setType(self.MIME)
            self.parcelable = ArrayList()
            for uri in uri_list:
                self.parcelable.add(uri)
            self.send.putParcelableArrayListExtra(Intent.EXTRA_STREAM,
                                                  self.parcelable)
            self.send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            if app:
                self.send.setPackage(app)
                mActivity.startActivity(self.send)
            else:
                self.choose = Intent.createChooser(self.send,None)
                mActivity.startActivity(self.choose)
        except Exception as e:
            Logger.warning('ShareSheet().share_file_list()')
            Logger.warning(str(e))            

    ######################################################
    # Legacy MediaStore (Android < 10) interface
    # Used to create a content uri to enable a Share
    # The uri will be destroyed on return from the Share 
    ######################################################
    def _legacy_create_uri(self, shared_file_path):
        uri = None
        if api_version > 28:
            if type(shared_file_path) != str:
                uri =  cast('android.net.Uri',shared_file_path)
        elif shared_file_path != None:
            if type(shared_file_path) == str:
                if exists(shared_file_path):
                    # If display_name still around, should (!) never happen
                    self._legacy_destroy_uri(shared_file_path)
                    # create uri
                    file_name = basename(shared_file_path)
                    self.display_name = file_name
                    self.MIME_type =\
                        SharedStorage().get_file_MIME_type(file_name)
                    cv = ContentValues()
                    cv.put(MediaStoreMediaColumns.DISPLAY_NAME,
                           self.display_name)
                    cv.put(MediaStoreMediaColumns.MIME_TYPE,
                           self.MIME_type)
                    cv.put(MediaStoreMediaColumns.DATA, shared_file_path)
                    root_uri = MediaStoreFiles.getContentUri('external')
                    context = mActivity.getApplicationContext()
                    uri = context.getContentResolver().insert(root_uri, cv)
                    uri = cast('android.net.Uri',uri)
                    self.legacy_uri_list.append(uri)
            else:
                uri = cast('android.net.Uri',shared_file_path)
        return uri

    # Destroy a previously created uri
    # The referenced file is unchanged,
    # and is still visible in the file system.
    def _cleanup_legacy_uri_list(self):
        resolver = mActivity.getApplicationContext().getContentResolver()
        for legacy_uri in self.legacy_uri_list:
            resolver.delete(legacy_uri,None,None)
        self.legacy_uri_list = []

    # This destroys a uri with the same display_name 
    # The referenced file is unchanged,
    # and is still visible in the file system.
    def _legacy_destroy_uri(self, shared_file_path):
        file_name = basename(shared_file_path)
        root_uri = MediaStoreFiles.getContentUri('external')
        self.selection = MediaStoreMediaColumns.DISPLAY_NAME+"=?" 
        self.args = [file_name]
        context = mActivity.getApplicationContext()
        cursor = context.getContentResolver().query(root_uri, None,
                                                    self.selection,
                                                    self.args, None)
        uri = None
        if cursor:
            while cursor.moveToNext():
                dn = MediaStoreMediaColumns.DISPLAY_NAME
                index = cursor.getColumnIndex(dn)
                cursor_file_name = cursor.getString(index)
                if file_name == cursor_file_name:
                    id_index = cursor.getColumnIndex('_id')
                    id = cursor.getLong(id_index)
                    uri = ContentUris.withAppendedId(root_uri,id)
                    break
            cursor.close()
        if uri:
            context.getContentResolver().delete(uri,None,None)
